#!/usr/bin/env python3
import requests
import sys
import os
import json
import argparse
from dotenv import load_dotenv

load_dotenv()

SERVER_URL = os.getenv("ANIMUS_SERVER_URL", "http://localhost:8080")
API_KEY = os.getenv("ANIMUS_API_KEY", "animus_dev_key")

headers = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

def get_status():
    try:
        response = requests.get(f"{SERVER_URL}/status", headers=headers)
        response.raise_for_status()
        data = response.json()
        if data["success"]:
            status = data["data"]
            print("\n--- Animus Farm Status ---")
            print(f"Running: {status['running']}")
            print(f"Paused: {status['paused']}")
            print(f"Current Stage: {status.get('current_stage', 'Idle')}")
            print(f"Current Video: {status.get('current_video_id', 'None')}")
            print(f"Videos Produced: {status['videos_produced']}")
            
            if status.get("next_scheduled_video"):
                print(f"Next Scheduled: {status['next_scheduled_video']}")
                print(f"Countdown: {status.get('hours_until_next', '?')} hours")
            
            if status.get("last_error"):
                print(f"Last Error: {status['last_error']}")
        else:
            print(f"Error: {data.get('error')}")
    except Exception as e:
        print(f"Connection failed: {e}")

def upload_script(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, 'r') as f:
        try:
            script_data = json.load(f)
        except json.JSONDecodeError:
            print("Invalid JSON file")
            return

    try:
        response = requests.post(f"{SERVER_URL}/manual/script", headers=headers, json=script_data)
        response.raise_for_status()
        print(response.json()["data"])
    except Exception as e:
        print(f"Upload failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Animus Farm Remote Control")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("status", help="Get current farm status")
    
    upload_parser = subparsers.add_parser("upload", help="Upload a manual script")
    upload_parser.add_argument("file", help="Path to JSON script file")

    args = parser.parse_args()
    
    if args.command == "status":
        get_status()
    elif args.command == "upload":
        upload_script(args.file)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
