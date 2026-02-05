#!/usr/bin/env python3
"""
MoviePy Bridge for Animus

Receives JSON configuration via stdin, processes video, outputs result to stdout.
"""

import json
import sys
import os
from typing import Optional

try:
    # MoviePy v2.x imports directly from moviepy
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
        from moviepy.video.fx.resize import resize
        from moviepy.video.fx.fadein import fadein
        from moviepy.video.fx.fadeout import fadeout
        MOVIEPY_V2 = False
    except ImportError:
        print(json.dumps({
            "success": False,
            "error": "MoviePy not installed. Run: pip install moviepy"
        }))
        sys.exit(1)


def apply_fade(clip, fade_in_duration=0.5, fade_out_duration=0.5):
    """Apply fade in/out effects, compatible with both MoviePy v1 and v2."""
    if MOVIEPY_V2:
        # MoviePy v2.x uses with_effects
        clip = clip.with_effects([vfx.FadeIn(fade_in_duration), vfx.FadeOut(fade_out_duration)])
    else:
        # MoviePy v1.x uses function calls
        clip = fadein(clip, fade_in_duration)
        clip = fadeout(clip, fade_out_duration)
    return clip


def create_section_video(
    section_assets: dict,
    start_time: float,
    end_time: float,
    width: int,
    height: int
) -> Optional[VideoFileClip]:
    """Create a video clip for a single section from available assets."""
    
    duration = end_time - start_time
    clips = []
    
    # Get available video clips for this section
    video_clips = section_assets.get("video_clips", [])
    
    if not video_clips:
        # Create a black clip if no assets
        return ColorClip(size=(width, height), color=(0, 0, 0), duration=duration)
    
    # Distribute clips across the section duration
    clip_duration = duration / max(len(video_clips), 1)
    current_time = 0
    
    for clip_info in video_clips:
        clip_path = clip_info.get("path", "")
        if not os.path.exists(clip_path):
            continue
            
        try:
            clip = VideoFileClip(clip_path)
            
            # Resize to match output dimensions (compatible with both versions)
            if MOVIEPY_V2:
                clip = clip.resized(height=height)
                if clip.w < width:
                    clip = clip.resized(width=width)
            else:
                clip = clip.resize(height=height)
                if clip.w < width:
                    clip = clip.resize(width=width)
            
            # Crop to exact dimensions
            x_center = clip.w / 2
            y_center = clip.h / 2
            if MOVIEPY_V2:
                clip = clip.cropped(
                    x_center=x_center, y_center=y_center,
                    width=width, height=height
                )
            else:
                clip = clip.crop(
                    x_center=x_center, y_center=y_center,
                    width=width, height=height
                )
            
            # Trim or loop to fit section
            if clip.duration > clip_duration:
                if MOVIEPY_V2:
                    clip = clip.subclipped(0, clip_duration)
                else:
                    clip = clip.subclip(0, clip_duration)
            elif clip.duration < clip_duration:
                # Loop the clip
                loops_needed = int(clip_duration / clip.duration) + 1
                looped = concatenate_videoclips([clip] * loops_needed)
                if MOVIEPY_V2:
                    clip = looped.subclipped(0, clip_duration)
                else:
                    clip = looped.subclip(0, clip_duration)
            
            # Add crossfade
            clip = apply_fade(clip, 0.5, 0.5)
            
            # Set start time
            if MOVIEPY_V2:
                clip = clip.with_start(current_time)
            else:
                clip = clip.set_start(current_time)
            
            clips.append(clip)
            current_time += clip_duration
            
        except Exception as e:
            print(f"Warning: Failed to process clip {clip_path}: {e}", file=sys.stderr)
            continue
    
    if not clips:
        return ColorClip(size=(width, height), color=(0, 0, 0), duration=duration)
    
    composite = CompositeVideoClip(clips, size=(width, height))
    if MOVIEPY_V2:
        return composite.with_duration(duration)
    else:
        return composite.set_duration(duration)


def assemble_video(config: dict) -> dict:
    """Main video assembly function."""
    
    video_id = config.get("video_id", "unknown")
    audio_path = config.get("audio_path")
    asset_manifest = config.get("asset_manifest", {})
    audio_timing = config.get("audio_timing", {})
    output_path = config.get("output_path")
    video_config = config.get("config", {})
    
    width = video_config.get("width", 1920)
    height = video_config.get("height", 1080)
    fps = video_config.get("fps", 30)
    
    if not audio_path or not os.path.exists(audio_path):
        return {"success": False, "error": f"Audio file not found: {audio_path}"}
    
    if not output_path:
        return {"success": False, "error": "No output path specified"}
    
    try:
        # Load audio
        audio = AudioFileClip(audio_path)
        total_duration = audio.duration
        
        # Get section timings
        section_timings = audio_timing.get("section_timings", [])
        section_assets_list = asset_manifest.get("section_assets", [])
        
        # Create video clips for each section
        video_clips = []
        
        for i, timing in enumerate(section_timings):
            start = timing.get("start_seconds", 0)
            end = timing.get("end_seconds", start + 60)
            
            # Get corresponding assets
            assets = section_assets_list[i] if i < len(section_assets_list) else {}
            
            section_clip = create_section_video(assets, start, end, width, height)
            if section_clip:
                if MOVIEPY_V2:
                    section_clip = section_clip.with_start(start)
                else:
                    section_clip = section_clip.set_start(start)
                video_clips.append(section_clip)
        
        # If no sections, create a black background
        if not video_clips:
            video_clips = [ColorClip(size=(width, height), color=(0, 0, 0), duration=total_duration)]
        
        # Composite all sections
        final_video = CompositeVideoClip(video_clips, size=(width, height))
        if MOVIEPY_V2:
            final_video = final_video.with_duration(total_duration)
        else:
            final_video = final_video.set_duration(total_duration)
        
        # Add audio
        if MOVIEPY_V2:
            final_video = final_video.with_audio(audio)
        else:
            final_video = final_video.set_audio(audio)
        
        # Write output
        final_video.write_videofile(
            output_path,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=f"/tmp/temp-audio-{video_id}.m4a",
            remove_temp=True,
            threads=4,
            preset="medium",
            verbose=False,
            logger=None
        )
        
        # Clean up
        final_video.close()
        audio.close()
        
        return {
            "success": True,
            "output_path": output_path,
            "duration_seconds": total_duration
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}


def main():
    """Read JSON from stdin, process, write JSON to stdout."""
    
    try:
        input_data = sys.stdin.read()
        config = json.loads(input_data)
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"Invalid JSON input: {e}"}))
        sys.exit(1)
    
    result = assemble_video(config)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
