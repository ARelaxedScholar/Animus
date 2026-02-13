#!/usr/bin/env python3
import os
import sys
import json
import requests
from dotenv import load_dotenv

# Load credentials from .env
load_dotenv()

CLIENT_ID = os.getenv("YOUTUBE_CLIENT_ID")
CLIENT_SECRET = os.getenv("YOUTUBE_CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("YOUTUBE_REFRESH_TOKEN")

if not CLIENT_ID or not CLIENT_SECRET or not REFRESH_TOKEN:
    print("Error: Missing YouTube credentials in .env")
    sys.exit(1)

def test_refresh_token():
    """Test if the refresh token is valid by attempting to get an access token."""
    # These should not be None due to earlier check, but satisfy type checker
    assert CLIENT_ID is not None
    assert REFRESH_TOKEN is not None
    
    print("Testing YouTube refresh token...")
    print(f"Client ID: {CLIENT_ID[:10]}...")
    print(f"Refresh token: {REFRESH_TOKEN[:10]}...")
    
    try:
        response = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "refresh_token": REFRESH_TOKEN,
                "grant_type": "refresh_token"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Refresh token is VALID!")
            print(f"Access token: {data.get('access_token', '')[:20]}...")
            print(f"Expires in: {data.get('expires_in')} seconds")
            print(f"Token type: {data.get('token_type')}")
            return True
        else:
            print(f"❌ Refresh token is INVALID (status {response.status_code})")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing token: {e}")
        return False

def test_youtube_api():
    """Test if we can access YouTube API with the token."""
    print("\nTesting YouTube API access...")
    
    try:
        # First get access token
        response = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "refresh_token": REFRESH_TOKEN,
                "grant_type": "refresh_token"
            },
            timeout=30
        )
        
        if response.status_code != 200:
            print("❌ Cannot get access token")
            return False
            
        access_token = response.json()["access_token"]
        
        # Test a simple API call (list channels)
        headers = {"Authorization": f"Bearer {access_token}"}
        api_response = requests.get(
            "https://www.googleapis.com/youtube/v3/channels",
            params={"part": "snippet", "mine": "true"},
            headers=headers,
            timeout=30
        )
        
        if api_response.status_code == 200:
            data = api_response.json()
            if "items" in data and len(data["items"]) > 0:
                channel = data["items"][0]["snippet"]
                print("✅ YouTube API access successful!")
                print(f"Channel: {channel.get('title')}")
                print(f"Description: {channel.get('description', '')[:100]}...")
                return True
            else:
                print("⚠️ API access OK but no channels found")
                return True
        else:
            print(f"❌ YouTube API error (status {api_response.status_code})")
            print(f"Error: {api_response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing YouTube API: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("YouTube Credentials Test")
    print("=" * 60)
    
    token_ok = test_refresh_token()
    
    if token_ok:
        api_ok = test_youtube_api()
        if api_ok:
            print("\n✅ All tests passed! YouTube integration is working.")
        else:
            print("\n⚠️ Token valid but API access failed. Check YouTube Data API v3 is enabled.")
    else:
        print("\n❌ Token validation failed. You need to obtain a new refresh token.")
        print("\nTo get a new refresh token:")
        print("1. Run: python scripts/get_youtube_tokens.py")
        print("2. Update .env with the new refresh token")
        print("3. Update database: just update-account-refresh 'Excelsior Academy' <new_token>")
    
    print("=" * 60)