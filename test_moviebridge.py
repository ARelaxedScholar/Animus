#!/usr/bin/env python3
"""Test the moviepy bridge with dummy assets."""
import json
import subprocess
import sys
import os
import tempfile
import shutil

def run_bridge(input_json: str, timeout=30):
    """Run the bridge with given JSON input."""
    bridge_script = "src/bridge/moviepy_bridge.py"
    # Use the same python environment
    result = subprocess.run(
        [sys.executable, bridge_script],
        input=input_json.encode(),
        capture_output=True,
        timeout=timeout
    )
    return result

def test_with_real_files():
    # Create temporary directory for test output
    test_dir = "/tmp/animus_test"
    os.makedirs(test_dir, exist_ok=True)
    
    # Paths to dummy assets (already created)
    audio_path = "/tmp/animus_test/assets/silent.mp3"
    clip_path = "/tmp/animus_test/assets/red.mp4"
    output_path = os.path.join(test_dir, "output.mp4")
    
    config = {
        "video_id": "test123",
        "audio_path": audio_path,
        "output_path": output_path,
        "mode": "horizontal",
        "config": {
            "width": 1920,
            "height": 1080,
            "fps": 30
        },
        "asset_manifest": {
            "video_id": "test123",
            "background_music": None,
            "section_assets": [
                {
                    "section_title": "Section 1",
                    "video_clips": [
                        {
                            "path": clip_path,
                            "source": "test",
                            "duration_seconds": 5.0,
                            "description": "red clip"
                        }
                    ],
                    "images": []
                }
            ]
        },
        "audio_timing": {
            "audio_path": audio_path,
            "total_duration_seconds": 5.0,
            "section_timings": [
                {
                    "section_title": "Section 1",
                    "start_seconds": 0.0,
                    "end_seconds": 5.0
                }
            ]
        }
    }
    
    input_json = json.dumps(config)
    print(f"Input JSON length: {len(input_json)}")
    
    try:
        result = run_bridge(input_json, timeout=60)  # 1 minute timeout
        print(f"Exit code: {result.returncode}")
        stdout = result.stdout.decode()
        stderr = result.stderr.decode()
        print(f"Stdout:\n{stdout}")
        if stderr:
            print(f"Stderr:\n{stderr}")
        
        # Parse JSON output
        try:
            output = json.loads(stdout)
            print(f"Parsed output: {json.dumps(output, indent=2)}")
            if output.get("success"):
                print("✅ Bridge succeeded!")
                if os.path.exists(output_path):
                    size = os.path.getsize(output_path)
                    print(f"Output video size: {size} bytes")
                else:
                    print("⚠️ Output video file not created")
            else:
                print("❌ Bridge reported failure")
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON output: {e}")
            print(f"Raw stdout: {stdout[:500]}")
    except subprocess.TimeoutExpired:
        print("❌ Bridge timed out after 60 seconds")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_with_real_files()