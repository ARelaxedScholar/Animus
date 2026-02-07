#!/usr/bin/env python3
"""
Analytics Worker for Animus DSPy Training Loop

This worker fetches YouTube Analytics data for published videos and computes
performance scores used to train the Judge (predictor) and Writer (generator).

The Harmonic Mean formula ensures balanced optimization:
    Score = 3 / (1/V_norm + 1/L_norm + 1/R_norm)

Where:
    V_norm = views / channel_avg_views (capped at 2.0)
    L_norm = like_ratio / channel_avg_like_ratio (capped at 2.0)  
    R_norm = retention_percentage / channel_avg_retention (capped at 2.0)

Run weekly via cron or manually via: python scripts/analytics_worker.py
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any

import psycopg2
from psycopg2.extras import RealDictCursor
import requests
from dotenv import load_dotenv

load_dotenv()

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
YOUTUBE_CLIENT_ID = os.getenv("YOUTUBE_CLIENT_ID")
YOUTUBE_CLIENT_SECRET = os.getenv("YOUTUBE_CLIENT_SECRET")
YOUTUBE_REFRESH_TOKEN = os.getenv("YOUTUBE_REFRESH_TOKEN")

# How many hours after publish before we fetch analytics (default: 7 days)
MIN_HOURS_SINCE_PUBLISH = int(os.getenv("ANALYTICS_MIN_HOURS", "168"))

# Cap for normalized values to prevent outliers from dominating
NORMALIZATION_CAP = 2.0


class YouTubeAnalytics:
    """Client for YouTube Analytics and Data APIs."""
    
    def __init__(self, client_id: str, client_secret: str, refresh_token: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.refresh_token = refresh_token
        self._access_token: Optional[str] = None
        self._token_expires: Optional[datetime] = None
    
    def _refresh_access_token(self) -> str:
        """Get a fresh access token using the refresh token."""
        response = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "refresh_token": self.refresh_token,
                "grant_type": "refresh_token",
            }
        )
        response.raise_for_status()
        data = response.json()
        self._access_token = data["access_token"]
        self._token_expires = datetime.now() + timedelta(seconds=data.get("expires_in", 3600) - 60)
        return self._access_token
    
    @property
    def access_token(self) -> str:
        """Get a valid access token, refreshing if needed."""
        if not self._access_token or not self._token_expires or datetime.now() >= self._token_expires:
            return self._refresh_access_token()
        return self._access_token
    
    def get_video_stats(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch basic video statistics from YouTube Data API.
        
        Returns: {view_count, like_count, comment_count} or None
        """
        try:
            response = requests.get(
                "https://www.googleapis.com/youtube/v3/videos",
                params={
                    "part": "statistics,contentDetails",
                    "id": video_id,
                },
                headers={"Authorization": f"Bearer {self.access_token}"}
            )
            response.raise_for_status()
            data = response.json()
            
            if not data.get("items"):
                print(f"  Warning: Video {video_id} not found in YouTube API")
                return None
            
            item = data["items"][0]
            stats = item.get("statistics", {})
            
            # Parse duration from contentDetails (ISO 8601 format: PT15M33S)
            duration_iso = item.get("contentDetails", {}).get("duration", "PT0S")
            duration_seconds = self._parse_duration(duration_iso)
            
            return {
                "view_count": int(stats.get("viewCount", 0)),
                "like_count": int(stats.get("likeCount", 0)),
                "comment_count": int(stats.get("commentCount", 0)),
                "duration_seconds": duration_seconds,
            }
        except Exception as e:
            print(f"  Error fetching stats for {video_id}: {e}")
            return None
    
    def get_video_retention(self, video_id: str, published_at: datetime) -> Optional[Dict[str, Any]]:
        """
        Fetch retention/watch time data from YouTube Analytics API.
        
        Note: YouTube Analytics API requires the channel to be linked and
        has a 2-day delay on data availability.
        
        Returns: {average_view_duration, average_view_percentage, retention_curve} or None
        """
        try:
            # Analytics API requires date range
            start_date = published_at.strftime("%Y-%m-%d")
            end_date = datetime.now().strftime("%Y-%m-%d")
            
            response = requests.get(
                "https://youtubeanalytics.googleapis.com/v2/reports",
                params={
                    "ids": "channel==MINE",
                    "startDate": start_date,
                    "endDate": end_date,
                    "metrics": "averageViewDuration,averageViewPercentage",
                    "dimensions": "video",
                    "filters": f"video=={video_id}",
                },
                headers={"Authorization": f"Bearer {self.access_token}"}
            )
            response.raise_for_status()
            data = response.json()
            
            rows = data.get("rows", [])
            if not rows:
                print(f"  Warning: No analytics data for {video_id} (may be too recent)")
                return None
            
            row = rows[0]
            return {
                "average_view_duration": row[1] if len(row) > 1 else None,
                "average_view_percentage": row[2] if len(row) > 2 else None,
            }
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                print(f"  Warning: Analytics API access denied for {video_id}")
                print("  Make sure 'youtube.readonly' scope is authorized and Analytics API is enabled.")
            else:
                print(f"  Error fetching analytics for {video_id}: {e}")
            return None
        except Exception as e:
            print(f"  Error fetching analytics for {video_id}: {e}")
            return None
    
    @staticmethod
    def _parse_duration(iso_duration: str) -> int:
        """Parse ISO 8601 duration (PT15M33S) to seconds."""
        import re
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', iso_duration)
        if not match:
            return 0
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        return hours * 3600 + minutes * 60 + seconds


def compute_harmonic_mean_score(
    views: int,
    like_ratio: float,
    retention_pct: float,
    baseline_views: float,
    baseline_like_ratio: float,
    baseline_retention: float,
) -> float:
    """
    Compute the harmonic mean of normalized metrics.
    
    The harmonic mean penalizes videos that fail in any single dimension,
    encouraging balanced performance across all metrics.
    """
    # Normalize each metric against channel baseline
    # Cap at NORMALIZATION_CAP to prevent outliers from dominating
    v_norm = min(views / max(baseline_views, 1), NORMALIZATION_CAP)
    l_norm = min(like_ratio / max(baseline_like_ratio, 0.001), NORMALIZATION_CAP)
    r_norm = min(retention_pct / max(baseline_retention, 1), NORMALIZATION_CAP)
    
    # Ensure no zeros (would make harmonic mean undefined)
    v_norm = max(v_norm, 0.01)
    l_norm = max(l_norm, 0.01)
    r_norm = max(r_norm, 0.01)
    
    # Harmonic mean: n / (1/x1 + 1/x2 + ... + 1/xn)
    harmonic_mean = 3.0 / (1.0/v_norm + 1.0/l_norm + 1.0/r_norm)
    
    # Scale to 0-1 range (harmonic mean of values capped at 2.0 will be at most 2.0)
    score = harmonic_mean / NORMALIZATION_CAP
    
    return min(max(score, 0.0), 1.0)


def get_or_compute_baseline(conn) -> Tuple[float, float, float]:
    """
    Get channel baseline metrics, or compute reasonable defaults.
    
    Returns: (avg_views, avg_like_ratio, avg_retention_pct)
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Try to get existing baseline
        cur.execute("""
            SELECT avg_views_30d, avg_likes_30d, avg_retention_30d
            FROM channel_baselines
            ORDER BY computed_at DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        
        if row and row["avg_views_30d"]:
            return (
                row["avg_views_30d"],
                row["avg_likes_30d"] or 0.05,  # Default 5% like ratio
                row["avg_retention_30d"] or 40.0,  # Default 40% retention
            )
        
        # No baseline exists - compute from existing videos
        cur.execute("""
            SELECT 
                AVG(vp.view_count) as avg_views,
                AVG(vp.like_ratio) as avg_like_ratio,
                AVG(vp.average_view_percentage) as avg_retention
            FROM video_performance vp
            JOIN videos v ON v.id = vp.video_id
            WHERE v.published_at > NOW() - INTERVAL '30 days'
        """)
        stats = cur.fetchone()
        
        if stats and stats["avg_views"]:
            return (
                float(stats["avg_views"]),
                float(stats["avg_like_ratio"] or 0.05),
                float(stats["avg_retention"] or 40.0),
            )
        
        # No data at all - use conservative defaults
        print("  No baseline data available, using defaults")
        return (100.0, 0.05, 40.0)


def process_video(
    conn,
    youtube: YouTubeAnalytics,
    video_id: str,
    youtube_video_id: str,
    published_at: datetime,
    baseline: Tuple[float, float, float],
) -> Optional[float]:
    """
    Fetch analytics for a single video and compute its performance score.
    
    Returns the computed score, or None if analytics unavailable.
    """
    print(f"  Processing video {video_id[:8]}... (YouTube: {youtube_video_id})")
    
    # Fetch basic stats
    stats = youtube.get_video_stats(youtube_video_id)
    if not stats:
        return None
    
    view_count = stats["view_count"]
    like_count = stats["like_count"]
    comment_count = stats["comment_count"]
    duration_seconds = stats["duration_seconds"]
    
    # Fetch retention data
    retention_data = youtube.get_video_retention(youtube_video_id, published_at)
    
    avg_view_duration = None
    avg_view_percentage = None
    
    if retention_data:
        avg_view_duration = retention_data.get("average_view_duration")
        avg_view_percentage = retention_data.get("average_view_percentage")
    elif duration_seconds > 0 and view_count > 0:
        # Estimate retention from duration if analytics unavailable
        # This is a rough heuristic - real data is better
        print(f"    Using estimated retention (no Analytics API data)")
        avg_view_percentage = 35.0  # Conservative estimate
    
    # Compute like ratio
    like_ratio = like_count / max(view_count, 1)
    
    # Calculate hours since publish
    hours_since_publish = int((datetime.now() - published_at.replace(tzinfo=None)).total_seconds() / 3600)
    
    # Insert performance snapshot
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO video_performance 
            (video_id, view_count, like_count, comment_count,
             average_view_duration_seconds, average_view_percentage,
             like_ratio, comment_ratio, hours_since_publish)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            video_id,
            view_count,
            like_count,
            comment_count,
            avg_view_duration,
            avg_view_percentage,
            like_ratio,
            comment_count / max(view_count, 1),
            hours_since_publish,
        ))
    
    # Compute harmonic mean score
    baseline_views, baseline_like_ratio, baseline_retention = baseline
    
    score = compute_harmonic_mean_score(
        views=view_count,
        like_ratio=like_ratio,
        retention_pct=avg_view_percentage or 35.0,
        baseline_views=baseline_views,
        baseline_like_ratio=baseline_like_ratio,
        baseline_retention=baseline_retention,
    )
    
    # Update video with performance score
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE videos 
            SET performance_score = %s, performance_updated_at = NOW() 
            WHERE id = %s
        """, (score, video_id))
    
    print(f"    Views: {view_count}, Likes: {like_count}, Retention: {avg_view_percentage or 'N/A'}%")
    print(f"    Score: {score:.3f}")
    
    return score


def update_channel_baseline(conn, youtube: YouTubeAnalytics):
    """
    Recompute channel baseline from recent video performance.
    Should be run weekly to keep normalization accurate.
    """
    print("\nUpdating channel baseline...")
    
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # 7-day averages
        cur.execute("""
            SELECT 
                AVG(vp.view_count) as avg_views,
                AVG(vp.like_ratio) as avg_like_ratio,
                AVG(vp.average_view_percentage) as avg_retention,
                COUNT(*) as video_count
            FROM video_performance vp
            JOIN videos v ON v.id = vp.video_id
            WHERE v.published_at > NOW() - INTERVAL '7 days'
              AND vp.fetched_at = (
                  SELECT MAX(fetched_at) FROM video_performance WHERE video_id = vp.video_id
              )
        """)
        stats_7d = cur.fetchone()
        
        # 30-day averages
        cur.execute("""
            SELECT 
                AVG(vp.view_count) as avg_views,
                AVG(vp.like_ratio) as avg_like_ratio,
                AVG(vp.average_view_percentage) as avg_retention,
                COUNT(*) as video_count
            FROM video_performance vp
            JOIN videos v ON v.id = vp.video_id
            WHERE v.published_at > NOW() - INTERVAL '30 days'
              AND vp.fetched_at = (
                  SELECT MAX(fetched_at) FROM video_performance WHERE video_id = vp.video_id
              )
        """)
        stats_30d = cur.fetchone()
        
        # Insert new baseline
        cur.execute("""
            INSERT INTO channel_baselines 
            (avg_views_7d, avg_likes_7d, avg_retention_7d,
             avg_views_30d, avg_likes_30d, avg_retention_30d,
             video_count_7d, video_count_30d)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            stats_7d["avg_views"] if stats_7d else None,
            stats_7d["avg_like_ratio"] if stats_7d else None,
            stats_7d["avg_retention"] if stats_7d else None,
            stats_30d["avg_views"] if stats_30d else None,
            stats_30d["avg_like_ratio"] if stats_30d else None,
            stats_30d["avg_retention"] if stats_30d else None,
            stats_7d["video_count"] if stats_7d else 0,
            stats_30d["video_count"] if stats_30d else 0,
        ))
        
    print(f"  7-day: {stats_7d['video_count'] if stats_7d else 0} videos, "
          f"avg views: {stats_7d['avg_views']:.0f if stats_7d and stats_7d['avg_views'] else 0}")
    print(f"  30-day: {stats_30d['video_count'] if stats_30d else 0} videos, "
          f"avg views: {stats_30d['avg_views']:.0f if stats_30d and stats_30d['avg_views'] else 0}")


def export_training_data(conn, output_path: str, min_score: Optional[float] = None):
    """
    Export training data for DSPy optimization.
    
    Generates a JSONL file with (topic_brief, script, score) tuples.
    """
    print(f"\nExporting training data to {output_path}...")
    
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        query = """
            SELECT id, topic_brief, script, performance_score
            FROM videos
            WHERE status = 'published'
              AND script IS NOT NULL
              AND performance_score IS NOT NULL
        """
        if min_score:
            query += f" AND performance_score >= {min_score}"
        query += " ORDER BY performance_score DESC"
        
        cur.execute(query)
        rows = cur.fetchall()
    
    with open(output_path, 'w') as f:
        for row in rows:
            record = {
                "video_id": str(row["id"]),
                "topic_brief": row["topic_brief"],
                "script": row["script"],
                "performance_score": float(row["performance_score"]),
            }
            f.write(json.dumps(record) + "\n")
    
    print(f"  Exported {len(rows)} training examples")


def main():
    parser = argparse.ArgumentParser(description="Animus Analytics Worker")
    parser.add_argument("--update-baseline", action="store_true", 
                       help="Update channel baseline metrics")
    parser.add_argument("--export", type=str, metavar="PATH",
                       help="Export training data to JSONL file")
    parser.add_argument("--min-score", type=float, default=None,
                       help="Minimum score for export (0.0-1.0)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be processed without making changes")
    parser.add_argument("--min-hours", type=int, default=MIN_HOURS_SINCE_PUBLISH,
                       help=f"Minimum hours since publish (default: {MIN_HOURS_SINCE_PUBLISH})")
    args = parser.parse_args()
    
    # Validate config
    if not DATABASE_URL:
        print("Error: DATABASE_URL not set in environment")
        sys.exit(1)
    
    if not all([YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, YOUTUBE_REFRESH_TOKEN]):
        print("Error: YouTube credentials not set in environment")
        print("Required: YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, YOUTUBE_REFRESH_TOKEN")
        sys.exit(1)
    
    # Connect to database
    print("Connecting to database...")
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    
    try:
        # Initialize YouTube client
        youtube = YouTubeAnalytics(
            YOUTUBE_CLIENT_ID,
            YOUTUBE_CLIENT_SECRET,
            YOUTUBE_REFRESH_TOKEN,
        )
        
        # Export mode
        if args.export:
            export_training_data(conn, args.export, args.min_score)
            return
        
        # Update baseline mode
        if args.update_baseline:
            update_channel_baseline(conn, youtube)
            conn.commit()
            return
        
        # Main processing mode: fetch analytics for videos needing updates
        print(f"\nFetching videos published > {args.min_hours} hours ago without scores...")
        
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, youtube_id, published_at
                FROM videos
                WHERE status = 'published'
                  AND youtube_id IS NOT NULL
                  AND published_at IS NOT NULL
                  AND published_at < NOW() - INTERVAL '1 hour' * %s
                  AND performance_score IS NULL
                ORDER BY published_at ASC
                LIMIT 50
            """, (args.min_hours,))
            videos = cur.fetchall()
        
        if not videos:
            print("No videos need performance updates.")
            return
        
        print(f"Found {len(videos)} videos to process")
        
        if args.dry_run:
            print("\n[DRY RUN] Would process:")
            for v in videos:
                print(f"  - {v['id'][:8]}... (YouTube: {v['youtube_id']})")
            return
        
        # Get baseline for normalization
        baseline = get_or_compute_baseline(conn)
        print(f"\nUsing baseline: views={baseline[0]:.0f}, like_ratio={baseline[1]:.3f}, retention={baseline[2]:.1f}%")
        
        # Process each video
        processed = 0
        for video in videos:
            try:
                score = process_video(
                    conn, youtube,
                    str(video["id"]),
                    video["youtube_id"],
                    video["published_at"],
                    baseline,
                )
                if score is not None:
                    processed += 1
                conn.commit()
            except Exception as e:
                print(f"  Error processing {video['id']}: {e}")
                conn.rollback()
        
        print(f"\nProcessed {processed}/{len(videos)} videos")
        
        # Update baseline if we processed videos
        if processed > 0:
            update_channel_baseline(conn, youtube)
            conn.commit()
        
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
