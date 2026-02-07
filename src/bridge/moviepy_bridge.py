#!/usr/bin/env python3
"""
MoviePy Bridge for Animus - Timeline Integrity Version 4

Fixes for the '1:00 freeze' issue:
1. Iterate over section_assets (not section_timings) to ensure all sections render
2. Use smart duration calculation instead of 60s fallback
3. Speed ramping for short B-roll clips (minimum 0.5x speed)
4. Validate video duration matches audio before final output
"""

import json
import sys
import os
import gc
import subprocess
from typing import Optional, List, Tuple

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
        from moviepy.video.fx.all import fadein, fadeout, speedx
        MOVIEPY_V2 = False
    except ImportError:
        print(json.dumps({
            "success": False,
            "error": "MoviePy not installed. Run: pip install moviepy"
        }))
        sys.exit(1)


# Minimum speed factor to avoid unnatural slow-motion
MIN_SPEED_FACTOR = 0.5


def apply_fade(clip, fade_in_duration=0.5, fade_out_duration=0.5):
    """Apply fade in/out effects."""
    # Ensure fade durations don't exceed clip duration
    clip_dur = clip.duration
    fade_in = min(fade_in_duration, clip_dur / 3)
    fade_out = min(fade_out_duration, clip_dur / 3)
    
    if MOVIEPY_V2:
        return clip.with_effects([vfx.FadeIn(fade_in), vfx.FadeOut(fade_out)])
    else:
        return fadeout(fadein(clip, fade_in), fade_out)


def apply_speed(clip, speed_factor: float):
    """Apply speed change to a clip. Speed < 1.0 = slower."""
    if MOVIEPY_V2:
        return clip.with_speed_scaled(speed_factor)
    else:
        return speedx(clip, speed_factor)


def extend_clip_with_speed(clip, target_duration: float) -> Tuple[any, float]:
    """
    Extend a clip using speed ramping (slowing down).
    Never goes below MIN_SPEED_FACTOR to avoid unnatural motion.
    
    Returns: (extended_clip, remaining_duration_to_fill)
    """
    actual_duration = clip.duration
    
    if actual_duration >= target_duration:
        # Clip is long enough, just trim
        if MOVIEPY_V2:
            return clip.subclipped(0, target_duration), 0
        else:
            return clip.subclip(0, target_duration), 0
    
    # Calculate how much we can extend by slowing down
    max_extended = actual_duration / MIN_SPEED_FACTOR  # e.g., 10s / 0.5 = 20s max
    
    if max_extended >= target_duration:
        # We can fill entirely with speed ramping
        speed = actual_duration / target_duration  # e.g., 10/15 = 0.67x speed
        extended = apply_speed(clip, speed)
        return extended, 0
    else:
        # Slow down to max, return remaining gap
        slowed = apply_speed(clip, MIN_SPEED_FACTOR)
        remaining = target_duration - max_extended
        return slowed, remaining


def write_video(clip, path, fps, **kwargs):
    """Write an intermediate video file with fixed settings for concatenation."""
    write_args = {
        "fps": fps,
        "codec": "libx264",
        "threads": 2,
        "audio": False,
        "preset": "ultrafast",
        "ffmpeg_params": [
            "-pix_fmt", "yuv420p",
            "-r", str(fps)  # Force constant framerate
        ]
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
    """
    Create a video file for a single section with intelligent B-roll handling.
    
    Uses speed ramping to extend short clips, maintaining visual interest
    while ensuring the section fills its required duration.
    """
    
    section_path = os.path.join(temp_dir, f"section_{section_index}.mp4")
    
    # Check if already exists (resume)
    if os.path.exists(section_path) and os.path.getsize(section_path) > 1000:
        print(f"  Section {section_index} already exists, skipping render.", file=sys.stderr)
        return section_path

    print(f"  Processing section {section_index} ({duration:.1f}s)...", file=sys.stderr)
    
    video_clips = section_assets.get("video_clips", [])
    
    if not video_clips:
        print(f"    No clips for section {section_index}, using black.", file=sys.stderr)
        clip = ColorClip(size=(width, height), color=(0, 0, 0), duration=duration)
        write_video(clip, section_path, fps=30)
        clip.close()
        return section_path
    
    # Calculate total available footage
    loaded_clips = []
    total_available = 0
    
    for clip_info in video_clips:
        path = clip_info.get("path", "")
        if not os.path.exists(path):
            print(f"    Clip not found: {path}", file=sys.stderr)
            continue
        
        try:
            clip = VideoFileClip(path, audio=False)
            
            # Resize to target dimensions
            if MOVIEPY_V2:
                clip = clip.resized(height=height)
                if clip.w < width:
                    clip = clip.resized(width=width)
                clip = clip.cropped(x_center=clip.w/2, y_center=clip.h/2, width=width, height=height)
            else:
                clip = clip.resize(height=height)
                if clip.w < width:
                    clip = clip.resize(width=width)
                clip = clip.crop(x_center=clip.w/2, y_center=clip.h/2, width=width, height=height)
            
            loaded_clips.append(clip)
            total_available += clip.duration
            
        except Exception as e:
            print(f"    Warning: Failed to load clip {path}: {e}", file=sys.stderr)
            continue
    
    if not loaded_clips:
        print(f"    All clips failed to load for section {section_index}, using black.", file=sys.stderr)
        clip = ColorClip(size=(width, height), color=(0, 0, 0), duration=duration)
        write_video(clip, section_path, fps=30)
        clip.close()
        return section_path
    
    print(f"    Loaded {len(loaded_clips)} clips, {total_available:.1f}s available, need {duration:.1f}s", file=sys.stderr)
    
    # Calculate speed factor to make clips fill the duration
    # If we have more footage than needed, we'll trim
    # If we have less, we'll slow down (up to MIN_SPEED_FACTOR limit)
    
    if total_available >= duration:
        # We have enough footage - distribute evenly and trim
        speed_factor = 1.0
        per_clip_duration = duration / len(loaded_clips)
    else:
        # We need to slow down clips
        # Calculate required speed to fill duration
        required_speed = total_available / duration
        
        if required_speed >= MIN_SPEED_FACTOR:
            # We can fill with speed ramping alone
            speed_factor = required_speed
            per_clip_duration = duration / len(loaded_clips)
            print(f"    Applying {speed_factor:.2f}x speed to fill duration", file=sys.stderr)
        else:
            # Even max slowdown won't fill - use max slowdown + will be short
            speed_factor = MIN_SPEED_FACTOR
            max_possible_duration = total_available / MIN_SPEED_FACTOR
            per_clip_duration = max_possible_duration / len(loaded_clips)
            print(f"    Warning: Even at {MIN_SPEED_FACTOR}x speed, can only fill {max_possible_duration:.1f}s of {duration:.1f}s", file=sys.stderr)
    
    # Build the composite
    active_clips = []
    current_start = 0
    
    for i, clip in enumerate(loaded_clips):
        try:
            # Apply speed factor
            if speed_factor != 1.0:
                clip = apply_speed(clip, speed_factor)
            
            # Calculate this clip's actual duration after speed change
            clip_actual_duration = clip.duration
            
            # Take what we need (or all if clip is shorter)
            take_duration = min(clip_actual_duration, per_clip_duration)
            
            if MOVIEPY_V2:
                clip = clip.subclipped(0, take_duration)
                clip = clip.with_start(current_start)
            else:
                clip = clip.subclip(0, take_duration)
                clip = clip.set_start(current_start)
            
            clip = apply_fade(clip)
            active_clips.append(clip)
            current_start += take_duration
            
        except Exception as e:
            print(f"    Warning: Failed to process clip {i}: {e}", file=sys.stderr)
            continue
    
    if not active_clips:
        print(f"    All clips failed processing for section {section_index}, using black.", file=sys.stderr)
        clip = ColorClip(size=(width, height), color=(0, 0, 0), duration=duration)
        write_video(clip, section_path, fps=30)
        clip.close()
        return section_path
    
    # Calculate actual composite duration
    actual_composite_duration = current_start
    
    # Create composite
    composite = CompositeVideoClip(active_clips, size=(width, height))
    
    # If we're short, we need to extend with the last frame or black
    if actual_composite_duration < duration:
        gap = duration - actual_composite_duration
        print(f"    Filling {gap:.1f}s gap with freeze frame", file=sys.stderr)
        
        # Create a freeze frame from the last clip's last frame
        try:
            last_clip = active_clips[-1]
            if MOVIEPY_V2:
                last_frame_time = last_clip.start + last_clip.duration - 0.1
                freeze = composite.to_ImageClip(t=max(0, last_frame_time))
                freeze = freeze.with_duration(gap).with_start(actual_composite_duration)
            else:
                last_frame_time = last_clip.start + last_clip.duration - 0.1
                freeze = composite.to_ImageClip(t=max(0, last_frame_time))
                freeze = freeze.set_duration(gap).set_start(actual_composite_duration)
            
            active_clips.append(freeze)
            composite = CompositeVideoClip(active_clips, size=(width, height))
        except Exception as e:
            print(f"    Warning: Could not create freeze frame: {e}", file=sys.stderr)
    
    # Set final duration
    if MOVIEPY_V2:
        composite = composite.with_duration(duration)
    else:
        composite = composite.set_duration(duration)
    
    write_video(composite, section_path, fps=30)
    
    # Cleanup
    composite.close()
    for c in loaded_clips:
        try:
            c.close()
        except:
            pass
    
    gc.collect()
    return section_path


def assemble_video(config: dict) -> dict:
    """
    Main video assembly function using FFmpeg filter_complex.
    
    Key fixes in v4:
    - Iterates over section_assets to ensure all sections render
    - Falls back to even duration distribution if timing data is missing
    - Validates output video duration matches audio
    """
    
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
    
    # Resume check for final video
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        print(f"Video already exists at {output_path}, checking validity...", file=sys.stderr)
        try:
            existing_video = VideoFileClip(output_path)
            dur = existing_video.duration
            existing_video.close()
            
            # Load audio to compare durations
            if audio_path and os.path.exists(audio_path):
                audio_check = AudioFileClip(audio_path)
                audio_dur = audio_check.duration
                audio_check.close()
                
                # Check if video is at least 95% of audio duration
                if dur >= audio_dur * 0.95:
                    print(f"Existing video is valid ({dur:.1f}s vs {audio_dur:.1f}s audio)", file=sys.stderr)
                    return {"success": True, "output_path": output_path, "duration_seconds": dur}
                else:
                    print(f"Existing video is too short ({dur:.1f}s vs {audio_dur:.1f}s audio), re-rendering...", file=sys.stderr)
            else:
                return {"success": True, "output_path": output_path, "duration_seconds": dur}
        except Exception as e:
            print(f"Existing video is corrupt ({e}), re-rendering...", file=sys.stderr)

    if not audio_path or not os.path.exists(audio_path):
        return {"success": False, "error": f"Audio file not found: {audio_path}"}
    
    try:
        print("Loading audio...", file=sys.stderr)
        audio = AudioFileClip(audio_path)
        total_duration = audio.duration
        print(f"Actual audio duration: {total_duration:.2f}s", file=sys.stderr)
        
        section_timings = audio_timing.get("section_timings", [])
        section_assets_list = asset_manifest.get("section_assets", [])
        
        print(f"Section timings: {len(section_timings)}, Section assets: {len(section_assets_list)}", file=sys.stderr)
        
        # CRITICAL FIX: Use section_assets as the source of truth for section count
        # This ensures we render ALL sections even if timing data is incomplete
        num_sections = len(section_assets_list)
        
        if num_sections == 0:
            return {"success": False, "error": "No section assets provided"}
        
        # Calculate section durations
        section_durations = []
        
        if len(section_timings) >= num_sections:
            # We have timing data for all sections - use it with scaling
            estimated_total = audio_timing.get("total_duration_seconds", 0)
            if estimated_total <= 0:
                estimated_total = sum(
                    (t.get("end_seconds", 0) - t.get("start_seconds", 0)) 
                    for t in section_timings
                )
            
            scale_factor = total_duration / estimated_total if estimated_total > 0 else 1.0
            print(f"Scale factor: {scale_factor:.4f} (actual {total_duration:.1f}s / est {estimated_total:.1f}s)", file=sys.stderr)
            
            for i in range(num_sections):
                timing = section_timings[i]
                start = timing.get("start_seconds", 0)
                end = timing.get("end_seconds", start + 10)  # 10s fallback, not 60s
                duration = (end - start) * scale_factor
                
                if duration < 0.5:
                    duration = total_duration / num_sections  # Fallback to even distribution
                    
                section_durations.append(duration)
        else:
            # Missing or incomplete timing data - distribute evenly
            print(f"Warning: Only {len(section_timings)} timings for {num_sections} sections, using even distribution", file=sys.stderr)
            even_duration = total_duration / num_sections
            section_durations = [even_duration] * num_sections
        
        # Normalize durations to match total audio duration exactly
        duration_sum = sum(section_durations)
        if abs(duration_sum - total_duration) > 0.1:
            adjustment = total_duration / duration_sum
            section_durations = [d * adjustment for d in section_durations]
            print(f"Adjusted section durations to match audio (factor: {adjustment:.4f})", file=sys.stderr)
        
        # Log section breakdown
        for i, dur in enumerate(section_durations):
            print(f"  Section {i}: {dur:.1f}s", file=sys.stderr)
        
        section_files = []
        
        # Step 1: Render individual sections
        for i in range(num_sections):
            duration = section_durations[i]
            assets = section_assets_list[i] if i < len(section_assets_list) else {}
            
            section_file = create_section_video(assets, duration, width, height, i, temp_dir)
            section_files.append(section_file)
        
        if not section_files:
            return {"success": False, "error": "No sections were rendered"}

        # Step 2: Use FFmpeg filter_complex to concatenate and mux
        print("Stitching sections and adding audio with FFmpeg...", file=sys.stderr)
        
        cmd = ["ffmpeg", "-y"]
        for sf in section_files:
            cmd.extend(["-i", sf])
        cmd.extend(["-i", audio_path])
        
        num_files = len(section_files)
        filter_complex = "".join([f"[{i}:v]" for i in range(num_files)])
        filter_complex += f"concat=n={num_files}:v=1:a=0[v]"
        
        cmd.extend([
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", f"{num_files}:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",  # Use shortest stream (in case video is slightly longer)
            "-movflags", "+faststart",
            output_path
        ])
        
        print(f"FFmpeg command: {' '.join(cmd[:10])}...", file=sys.stderr)
        result = subprocess.run(cmd, capture_output=True)
        
        if result.returncode != 0:
            error_msg = result.stderr.decode()
            print(f"FFmpeg failed: {error_msg}", file=sys.stderr)
            return {"success": False, "error": f"FFmpeg error: {error_msg[:500]}"}
        
        # Step 3: Validate output
        print("Validating output video...", file=sys.stderr)
        try:
            output_video = VideoFileClip(output_path)
            output_duration = output_video.duration
            output_video.close()
            
            duration_diff = abs(output_duration - total_duration)
            if duration_diff > 1.0:
                print(f"Warning: Output duration mismatch: {output_duration:.1f}s vs {total_duration:.1f}s audio", file=sys.stderr)
            else:
                print(f"Output validated: {output_duration:.1f}s (audio: {total_duration:.1f}s)", file=sys.stderr)
                
        except Exception as e:
            print(f"Warning: Could not validate output: {e}", file=sys.stderr)
            output_duration = total_duration
        
        # Step 4: Cleanup
        audio.close()
        for f in section_files:
            try:
                os.remove(f)
            except:
                pass
        
        return {
            "success": True,
            "output_path": output_path,
            "duration_seconds": output_duration
        }
        
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        print(f"FFmpeg failed: {error_msg}", file=sys.stderr)
        return {"success": False, "error": f"FFmpeg error: {error_msg[:500]}"}
    except Exception as e:
        import traceback
        print(traceback.format_exc(), file=sys.stderr)
        return {"success": False, "error": str(e)}


def main():
    """Read JSON from stdin, process, write JSON to stdout."""
    true_stdout = sys.stdout
    sys.stdout = sys.stderr

    print("MoviePy Bridge v4 starting...", file=sys.stderr)
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
