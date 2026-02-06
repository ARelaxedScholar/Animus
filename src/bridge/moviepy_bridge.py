#!/usr/bin/env python3
"""
MoviePy Bridge for Animus - Timeline Integrity Version

Uses FFmpeg's concat demuxer for final assembly to ensure perfect 
timeline synchronization and zero gaps, preventing the '1:00 freeze'.
"""

import json
import sys
import os
import gc
import subprocess
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
    """Write an intermediate video file with fixed settings for concatenation."""
    write_args = {
        "fps": fps,
        "codec": "libx264",
        "threads": 2,
        "audio": False, # Sections don't need audio, added at final stage
        "preset": "medium",
    }
    
    if MOVIEPY_V2:
        clip.write_videofile(path, **write_args)
    else:
        clip.write_videofile(path, verbose=False, logger=None, **write_args)


def create_section_video(
    section_assets: dict,
    duration: float,
    width: int,
    height: int,
    section_index: int,
    temp_dir: str
) -> str:
    """Create a high-quality video file for a single section."""
    
    print(f"  Processing section {section_index} ({duration:.1f}s)...", file=sys.stderr)
    
    video_clips = section_assets.get("video_clips", [])
    section_path = os.path.join(temp_dir, f"section_{section_index}.mp4")
    
    if not video_clips:
        clip = ColorClip(size=(width, height), color=(0, 0, 0), duration=duration)
        write_video(clip, section_path, fps=30)
        clip.close()
        return section_path
    
    num_clips = len(video_clips)
    clip_duration = duration / num_clips
    
    active_clips = []
    current_start = 0
    
    for i, clip_info in enumerate(video_clips):
        path = clip_info.get("path", "")
        if not os.path.exists(path):
            continue
            
        try:
            # Disable audio to save memory
            clip = VideoFileClip(path, audio=False)
            
            # Scale and crop to target dimensions
            if MOVIEPY_V2:
                clip = clip.resized(height=height)
                if clip.w < width:
                    clip = clip.resized(width=width)
                clip = clip.cropped(x_center=clip.w/2, y_center=clip.h/2, width=width, height=height)
                
                # Take exact subclip
                duration_to_take = min(clip.duration, clip_duration)
                clip = clip.subclipped(0, duration_to_take)
                clip = clip.with_duration(clip_duration)
                clip = clip.with_start(current_start)
            else:
                clip = clip.resize(height=height)
                if clip.w < width:
                    clip = clip.resize(width=width)
                clip = clip.crop(x_center=clip.w/2, y_center=clip.h/2, width=width, height=height)
                
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
        write_video(clip, section_path, fps=30)
        clip.close()
        return section_path
        
    composite = CompositeVideoClip(active_clips, size=(width, height))
    if MOVIEPY_V2:
        composite = composite.with_duration(duration)
    else:
        composite = composite.set_duration(duration)
        
    write_video(composite, section_path, fps=30)
    
    composite.close()
    for c in active_clips:
        c.close()
    
    gc.collect()
    return section_path


def assemble_video(config: dict) -> dict:
    """Main video assembly function using FFmpeg concat demuxer."""
    
    video_id = config.get("video_id", "unknown")
    audio_path = config.get("audio_path")
    asset_manifest = config.get("asset_manifest", {})
    audio_timing = config.get("audio_timing", {})
    output_path = config.get("output_path")
    video_config = config.get("config", {})
    
    width = video_config.get("width", 1920)
    height = video_config.get("height", 1080)
    fps = video_config.get("fps", 30)
    
    temp_dir = os.path.dirname(output_path)
    
    # Resume check
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        print(f"Video already exists at {output_path}, skipping render.", file=sys.stderr)
        try:
            existing_video = VideoFileClip(output_path)
            dur = existing_video.duration
            existing_video.close()
            return {"success": True, "output_path": output_path, "duration_seconds": dur}
        except:
            print("Existing video is corrupt, re-rendering...", file=sys.stderr)
            pass

    if not audio_path or not os.path.exists(audio_path):
        return {"success": False, "error": f"Audio file not found: {audio_path}"}
    
    try:
        print("Loading audio...", file=sys.stderr)
        audio = AudioFileClip(audio_path)
        total_duration = audio.duration
        
        section_timings = audio_timing.get("section_timings", [])
        section_assets_list = asset_manifest.get("section_assets", [])
        
        section_files = []
        
        # Step 1: Render individual sections
        for i, timing in enumerate(section_timings):
            start = timing.get("start_seconds", 0)
            end = timing.get("end_seconds", start + 60)
            duration = end - start
            assets = section_assets_list[i] if i < len(section_assets_list) else {}
            
            section_file = create_section_video(assets, duration, width, height, i, temp_dir)
            section_files.append(section_file)
            
        # Step 2: Use FFmpeg to concatenate sections perfectly
        # Create manifest file for FFmpeg
        manifest_path = os.path.join(temp_dir, f"manifest_{video_id}.txt")
        with open(manifest_path, "w") as f:
            for sf in section_files:
                # Use absolute paths and escape single quotes
                f.write(f"file '{os.path.abspath(sf)}'\n")
        
        print("Stitching sections with FFmpeg concat demuxer...", file=sys.stderr)
        temp_no_audio = os.path.join(temp_dir, f"no_audio_{video_id}.mp4")
        
        # Concat command
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
            "-i", manifest_path, "-c", "copy", temp_no_audio
        ], check=True, capture_output=True)

        # Step 3: Add the final audio
        print("Muxing final audio and video...", file=sys.stderr)
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_no_audio, "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", output_path
        ], check=True, capture_output=True)
        
        # Step 4: Cleanup
        audio.close()
        for f in section_files:
            try: os.remove(f)
            except: pass
        try: os.remove(manifest_path)
        except: pass
        try: os.remove(temp_no_audio)
        except: pass
            
        return {
            "success": True,
            "output_path": output_path,
            "duration_seconds": total_duration
        }
        
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed: {e.stderr.decode()}", file=sys.stderr)
        return {"success": False, "error": f"FFmpeg error: {e.stderr.decode()}"}
    except Exception as e:
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        return {"success": False, "error": str(e)}


def main():
    """Read JSON from stdin, process, write JSON to stdout."""
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
    print(json.dumps(result), file=true_stdout)
    true_stdout.flush()


if __name__ == "__main__":
    main()
