#!/usr/bin/env python3
"""
Add a new YouTube account to the database using OAuth2.

Usage:
    python scripts/add_youtube_account.py "Account Name" "niche"
"""

import os
import sys
import argparse
from dotenv import load_dotenv
import psycopg2
from google_auth_oauthlib.flow import InstalledAppFlow

# Load environment variables
load_dotenv()

def check_prerequisites(client_id):
    """Print prerequisite checks."""
    import time
    print("=" * 60)
    print("YouTube Account Setup Checklist:")
    print("=" * 60)
    print("1. Google Cloud Console:")
    print(f"   - Project has YouTube Data API v3 enabled")
    print(f"   - OAuth 2.0 Client ID: {client_id[:10]}...")
    print("   - Authorized redirect URIs includes: http://localhost")
    print("2. .env file has YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET")
    print("3. PostgreSQL is running (docker-compose up -d)")
    print("=" * 60)
    print("If OAuth fails, check the above items.")
    print("Continuing in 3 seconds...")
    time.sleep(3)

def get_refresh_token(client_id, client_secret):
    """Get a refresh token via OAuth2 flow."""
    SCOPES = [
        'https://www.googleapis.com/auth/youtube.upload',
        'https://www.googleapis.com/auth/youtube.readonly'
    ]
    
    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    
    print("\n🚀 Starting YouTube OAuth flow...")
    print("A browser window will open for authentication.")
    print("If no browser opens, please visit the URL shown below.")
    print("Make sure you're logged into the Google account for the YouTube channel you want to add.")
    
    try:
        flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
        
        # Use console flow with browser opening
        credentials = flow.run_local_server(
            port=0,
            authorization_prompt_message='Please visit this URL to authorize the app: {url}',
            success_message='The auth flow is complete; you may close this window.',
            open_browser=True
        )
        
        if not credentials.refresh_token:
            print("ERROR: No refresh token returned. Make sure you grant offline access.")
            print("Tip: When authorizing, make sure to check 'See, edit, create, and delete your YouTube videos'")
            sys.exit(1)
        
        return credentials.refresh_token
        
    except Exception as e:
        print(f"\n❌ OAuth Error: {e}")
        print("\nCommon fixes:")
        print("1. Make sure 'http://localhost' is added as authorized redirect URI in Google Cloud Console")
        print("2. Ensure YouTube Data API v3 is enabled for your project")
        print("3. Check that client_id and client_secret are correct")
        sys.exit(1)

def insert_account(db_url, name, niche, client_id, client_secret, refresh_token):
    """Insert or update account in database."""
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        
        # Check if account already exists
        cur.execute("SELECT id FROM youtube_accounts WHERE name = %s", (name,))
        existing = cur.fetchone()
        
        if existing:
            print(f"⚠️ Account '{name}' already exists. Updating...")
            cur.execute("""
                UPDATE youtube_accounts 
                SET niche = %s, client_id = %s, client_secret = %s, 
                    refresh_token = %s, updated_at = NOW(), is_active = true
                WHERE name = %s
                RETURNING id
            """, (niche, client_id, client_secret, refresh_token, name))
        else:
            cur.execute("""
                INSERT INTO youtube_accounts 
                (name, niche, client_id, client_secret, refresh_token, is_active)
                VALUES (%s, %s, %s, %s, %s, true)
                RETURNING id
            """, (name, niche, client_id, client_secret, refresh_token))
        
        row = cur.fetchone()
        if not row:
            print("ERROR: No ID returned from database insert")
            sys.exit(1)
        account_id = row[0]
        conn.commit()
        cur.close()
        conn.close()
        
        return account_id
        
    except Exception as e:
        print(f"Database error: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Add YouTube account to database')
    parser.add_argument('name', help='Account name (e.g., "Excelsior Academy")')
    parser.add_argument('niche', help='Niche (e.g., "stoicism", "philosophy")')
    parser.add_argument('--client-id', help='Google OAuth Client ID (defaults to YOUTUBE_CLIENT_ID from .env)')
    parser.add_argument('--client-secret', help='Google OAuth Client Secret (defaults to YOUTUBE_CLIENT_SECRET from .env)')
    
    args = parser.parse_args()
    
    # Get credentials from .env or arguments
    client_id = args.client_id or os.getenv("YOUTUBE_CLIENT_ID")
    client_secret = args.client_secret or os.getenv("YOUTUBE_CLIENT_SECRET")
    db_url = os.getenv("DATABASE_URL")
    
    if not client_id or not client_secret:
        print("ERROR: YouTube client credentials not found.")
        print("Set YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET in .env or provide via --client-id/--client-secret")
        sys.exit(1)
    
    if not db_url:
        print("ERROR: DATABASE_URL not found in .env")
        sys.exit(1)
    
    print(f"Adding YouTube account: {args.name}")
    print(f"Niche: {args.niche}")
    print(f"Client ID: {client_id[:10]}...")
    
    # Check prerequisites
    check_prerequisites(client_id)
    
    # Get refresh token via OAuth
    refresh_token = get_refresh_token(client_id, client_secret)
    
    print(f"\n✅ Refresh token obtained: {refresh_token[:20]}...")
    
    # Insert into database
    account_id = insert_account(db_url, args.name, args.niche, client_id, client_secret, refresh_token)
    
    print(f"\n🎉 Success! Account added with ID: {account_id}")
    print(f"Name: {args.name}")
    print(f"Niche: {args.niche}")
    print("\nYou can now use this account in Animus for video production.")
    print("Run 'just list-accounts' to verify.")

if __name__ == "__main__":
    main()