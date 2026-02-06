#!/usr/bin/env python3
"""
MoviePy Bridge for Animus - Memory Efficient Version

Assembles long-form video by processing sections independently and 
concatenating the results, avoiding keeping many clips open at once.
"""

import json
import sys
import os
import gc
from typing import Optional, List

try:
    # MoviePy v2.x imports
    from moviepy import (
        VideoFileClip, AudioFileClip, ImageClip, 
        CompositeVideoClip, concatenate_videoclips,
        ColorClip, TextClip
    )
    from moviepy import vfx
    MOVIEPY_V2 = True
except ImportError:
    try:
        # Fallback to MoviePy v1.x imports
        from moviepy.editor import (
            VideoFileClip, AudioFileClip, ImageClip, 
            CompositeVideoClip, concatenate_videoclips,
            ColorClip, TextClip
        )
        from moviepy.video.fx.all import fadein, fadeout
        MOVIEPY_V2 = False
    except ImportError:
        print(json.dumps({
            "success": False,
            "error": "MoviePy not installed. Run: pip install moviepy"
        }))
        sys.exit(1)


def apply_fade(clip, fade_in_duration=0.5, fade_out_duration=0.5):
    """Apply fade in/out effects."""
    if MOVIEPY_V2:
        return clip.with_effects([vfx.FadeIn(fade_in_duration), vfx.FadeOut(fade_out_duration)])
    else:
        return fadeout(fadein(clip, fade_in_duration), fade_out_duration)


def write_video(clip, path, fps, **kwargs):
    """Write a video file using version-compatible arguments."""
    write_args = {
        "fps": fps,
        "codec": "libx264",
        "threads": 2, # Reduced to save memory
    }
    
    # Add optional args if they are provided
    for key in ["audio", "audio_codec", "temp_audiofile", "remove_temp", "preset"]:
        if key in kwargs:
            write_args[key] = kwargs[key]
            
    if MOVIEPY_V2:
        # MoviePy 2.x doesn't use verbose/logger in the same way
        clip.write_videofile(path, **write_args)
    else:
        # MoviePy 1.x supports verbose/logger
        clip.write_videofile(path, verbose=False, logger=None, **write_args)


def create_section_video(
    section_assets: dict,
    duration: float,
    width: int,
    height: int,
    section_index: int,
    temp_dir: str
) -> str:
    """Create a video file for a single section and return its path."""
    
    print(f"  Processing section {section_index} ({duration:.1f}s)...", file=sys.stderr)
    
    video_clips = section_assets.get("video_clips", [])
    section_path = os.path.join(temp_dir, f"section_{section_index}.mp4")
    
    if not video_clips:
        clip = ColorClip(size=(width, height), color=(0, 0, 0), duration=duration)
        write_video(clip, section_path, fps=24)
        clip.close()
        return section_path
    
    # Distribute clips across the section duration
    num_clips = len(video_clips)
    clip_duration = duration / num_clips
    
    active_clips = []
    current_start = 0
    
    for i, clip_info in enumerate(video_clips):
        path = clip_info.get("path", "")
        if not os.path.exists(path):
            continue
            
        try:
            print(f"    Opening clip {i}: {os.path.basename(path)}", file=sys.stderr)
            # Disable audio to save memory and avoid issues
            clip = VideoFileClip(path, audio=False)
            
            # CRITICAL: Resize to target dimensions IMMEDIATELY to save RAM
            # MoviePy keeps the full frame in memory otherwise
            if MOVIEPY_V2:
                clip = clip.resized(height=height)
                if clip.w < width:
                    clip = clip.resized(width=width)
                clip = clip.cropped(x_center=clip.w/2, y_center=clip.h/2, width=width, height=height)
                
                # Limit duration
                duration_to_take = min(clip.duration, clip_duration)
                clip = clip.subclipped(0, duration_to_take)
                clip = clip.with_duration(clip_duration)
                clip = clip.with_start(current_start)
            else:
                clip = clip.resize(height=height)
                if clip.w < width:
                    clip = clip.resize(width=width)
                clip = clip.crop(x_center=clip.w/2, y_center=clip.h/2, width=width, height=height)
                
                # Limit duration
                duration_to_take = min(clip.duration, clip_duration)
                clip = clip.subclip(0, duration_to_take)
                clip = clip.set_duration(clip_duration)
                clip = clip.set_start(current_start)
            
            clip = apply_fade(clip)
            active_clips.append(clip)
            current_start += clip_duration
            
        except Exception as e:
            print(f"    Warning: Failed to process clip {path}: {e}", file=sys.stderr)
            continue
            
    if not active_clips:
        clip = ColorClip(size=(width, height), color=(0, 0, 0), duration=duration)
        write_video(clip, section_path, fps=24)
        clip.close()
        return section_path
        
    # Create composite for this section
    composite = CompositeVideoClip(active_clips, size=(width, height))
    if MOVIEPY_V2:
        composite = composite.with_duration(duration)
    else:
        composite = composite.set_duration(duration)
        
    # Write section to file
    write_video(composite, section_path, fps=24, audio=False)
    
    # CRITICAL: Close all clips to free memory
    composite.close()
    for c in active_clips:
        c.close()
    
    # Explicit GC
    gc.collect()
    
    return section_path


def assemble_video(config: dict) -> dict:
    """Main video assembly function using chunked processing."""
    
    video_id = config.get("video_id", "unknown")
    audio_path = config.get("audio_path")
    asset_manifest = config.get("asset_manifest", {})
    audio_timing = config.get("audio_timing", {})
    output_path = config.get("output_path")
    video_config = config.get("config", {})
    
    width = video_config.get("width", 1920)
    height = video_config.get("height", 1080)
    fps = video_config.get("fps", 30)
    
    # Get local temp dir from config or use a default near the output
    temp_dir = os.path.dirname(output_path)
    
    # CHECK IF VIDEO ALREADY EXISTS (Optimized resume)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        print(f"Video already exists at {output_path}, skipping render.", file=sys.stderr)
        try:
            existing_video = VideoFileClip(output_path)
            dur = existing_video.duration
            existing_video.close()
            return {
                "success": True,
                "output_path": output_path,
                "duration_seconds": dur
            }
        except:
            print("Existing video is corrupt, re-rendering...", file=sys.stderr)
            pass

    if not audio_path or not os.path.exists(audio_path):
        return {"success": False, "error": f"Audio file not found: {audio_path}"}
    
    try:
        # Load audio to get total duration
        print("Loading audio...", file=sys.stderr)
        audio = AudioFileClip(audio_path)
        total_duration = audio.duration
        
        section_timings = audio_timing.get("section_timings", [])
        section_assets_list = asset_manifest.get("section_assets", [])
        
        section_files = []
        
        # Step 1: Process each section independently
        for i, timing in enumerate(section_timings):
            start = timing.get("start_seconds", 0)
            end = timing.get("end_seconds", start + 60)
            duration = end - start
            
            assets = section_assets_list[i] if i < len(section_assets_list) else {}
            
            section_file = create_section_video(assets, duration, width, height, i, temp_dir)
            section_files.append(section_file)
            
        # Step 2: Concatenate section files
        print("Concatenating sections...", file=sys.stderr)
        section_clips = [VideoFileClip(f) for f in section_files]
        
        final_video = concatenate_videoclips(section_clips, method="compose")
        
        if MOVIEPY_V2:
            final_video = final_video.with_audio(audio)
        else:
            final_video = final_video.set_audio(audio)
        
        print(f"Writing final video to {output_path}...", file=sys.stderr)
        write_video(
            final_video,
            output_path,
            fps=fps,
            audio=True,
            audio_codec="aac",
            temp_audiofile=os.path.join(temp_dir, f"temp-audio-{video_id}.m4a"),
            remove_temp=True,
            preset="medium"
        )
        
        # Step 3: Cleanup
        final_video.close()
        audio.close()
        for c in section_clips:
            c.close()
        for f in section_files:
            try: os.remove(f)
            except: pass
            
        return {
            "success": True,
            "output_path": output_path,
            "duration_seconds": total_duration
        }
        
    except Exception as e:
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        return {"success": False, "error": str(e)}


def main():
    """Read JSON from stdin, process, write JSON to stdout."""
    
    # Redirect all further stdout to stderr so only our JSON result goes to real stdout
    true_stdout = sys.stdout
    sys.stdout = sys.stderr

    print("Bridge starting...", file=sys.stderr)
    try:
        input_data = sys.stdin.read()
        if not input_data:
            print(json.dumps({"success": False, "error": "Empty input"}), file=true_stdout)
            sys.exit(1)
        config = json.loads(input_data)
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"Invalid JSON input: {e}"}), file=true_stdout)
        sys.exit(1)
    
    result = assemble_video(config)
    
    # Send result to the REAL stdout
    print(json.dumps(result), file=true_stdout)
    true_stdout.flush()


if __name__ == "__main__":
    main()
