#!/usr/bin/env python3
import os
import json
from google_auth_oauthlib.flow import InstalledAppFlow
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

CLIENT_ID = os.getenv("YOUTUBE_CLIENT_ID")
CLIENT_SECRET = os.getenv("YOUTUBE_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    print("Error: YOUTUBE_CLIENT_ID or YOUTUBE_CLIENT_SECRET not found in .env")
    exit(1)

# Define scopes
SCOPES = [
    'https://www.googleapis.com/auth/youtube.upload',
    'https://www.googleapis.com/auth/youtube.readonly'
]

def main():
    client_config = {
        "installed": {
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }

    # Run the flow
    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    
    # Use console flow since we are in a CLI environment
    credentials = flow.run_local_server(
        port=0,
        authorization_prompt_message='Please visit this URL to authorize the app: {url}',
        success_message='The auth flow is complete; you may close this window.',
        open_browser=False
    )

    print("\n--- NEW YOUTUBE REFRESH TOKEN ---")
    print(credentials.refresh_token)
    print("---------------------------------\n")
    print("Update your .env file with this token and restart the daemon.")

if __name__ == "__main__":
    main()
