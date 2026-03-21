#!/usr/bin/env python3
"""Fetch YouTube analytics, score videos, update baselines, and export training data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras
import requests
from dotenv import load_dotenv


YOUTUBE_VIDEOS_API = "https://www.googleapis.com/youtube/v3/videos"
OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"
DEFAULT_MIN_HOURS = 168
YOUTUBE_BATCH_SIZE = 50


class FatalSetupError(Exception):
    """Raised when setup/configuration errors should terminate execution."""


@dataclass
class Baseline:
    avg_views_30d: float
    avg_likes_30d: float
    avg_comment_ratio_30d: float


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def get_database_connection() -> psycopg2.extensions.connection:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise FatalSetupError("DATABASE_URL is required")

    try:
        return psycopg2.connect(database_url)
    except Exception as exc:
        raise FatalSetupError(f"Failed to connect to database: {exc}") from exc


def get_youtube_auth() -> tuple[dict[str, str], dict[str, str]]:
    api_key = os.getenv("YOUTUBE_API_KEY")
    if api_key:
        return {}, {"key": api_key}

    client_id = os.getenv("YOUTUBE_CLIENT_ID")
    client_secret = os.getenv("YOUTUBE_CLIENT_SECRET")
    refresh_token = os.getenv("YOUTUBE_REFRESH_TOKEN")
    if not client_id or not client_secret or not refresh_token:
        raise FatalSetupError(
            "YouTube auth missing. Set YOUTUBE_API_KEY or OAuth env vars "
            "(YOUTUBE_CLIENT_ID/YOUTUBE_CLIENT_SECRET/YOUTUBE_REFRESH_TOKEN)."
        )

    try:
        response = requests.post(
            OAUTH_TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        raise FatalSetupError(f"Failed to obtain YouTube OAuth access token: {exc}") from exc

    access_token = payload.get("access_token")
    if not access_token:
        raise FatalSetupError("OAuth token response did not include access_token")

    return {"Authorization": f"Bearer {access_token}"}, {}


def fetch_videos_needing_update(
    conn: psycopg2.extensions.connection, min_hours: int
) -> list[dict[str, Any]]:
    query = """
        SELECT id::text AS id, youtube_id, published_at
        FROM videos
        WHERE status = 'published'
          AND youtube_id IS NOT NULL
          AND published_at IS NOT NULL
          AND published_at < NOW() - INTERVAL '1 hour' * %s
          AND performance_score IS NULL
        ORDER BY published_at ASC
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, (min_hours,))
        return list(cur.fetchall())


def fetch_youtube_stats_batch(
    video_ids: list[str], headers: dict[str, str], auth_params: dict[str, str]
) -> dict[str, dict[str, int]]:
    params = {
        "part": "statistics",
        "id": ",".join(video_ids),
        **auth_params,
    }
    response = requests.get(YOUTUBE_VIDEOS_API, params=params, headers=headers, timeout=20)
    response.raise_for_status()
    payload = response.json()

    stats_by_video_id: dict[str, dict[str, int]] = {}
    for item in payload.get("items", []):
        video_id = item.get("id")
        stats = item.get("statistics", {})
        if not video_id:
            continue
        try:
            stats_by_video_id[video_id] = {
                "view_count": int(stats.get("viewCount", 0) or 0),
                "like_count": int(stats.get("likeCount", 0) or 0),
                "comment_count": int(stats.get("commentCount", 0) or 0),
            }
        except (TypeError, ValueError):
            continue

    return stats_by_video_id


def get_latest_baseline(conn: psycopg2.extensions.connection) -> Baseline:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT avg_views_30d, avg_likes_30d
            FROM channel_baselines
            ORDER BY computed_at DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()

        cur.execute(
            """
            WITH latest AS (
                SELECT DISTINCT ON (video_id) comment_ratio
                FROM video_performance
                WHERE fetched_at >= NOW() - INTERVAL '30 days'
                ORDER BY video_id, fetched_at DESC
            )
            SELECT AVG(comment_ratio)::float
            FROM latest
            WHERE comment_ratio IS NOT NULL
            """
        )
        comment_row = cur.fetchone()

    avg_views_30d = float((row[0] if row and row[0] is not None else 1000.0) or 1000.0)
    avg_likes_30d = float((row[1] if row and row[1] is not None else 50.0) or 50.0)
    avg_comment_ratio_30d = float(
        (comment_row[0] if comment_row and comment_row[0] is not None else 0.01) or 0.01
    )

    return Baseline(
        avg_views_30d=max(avg_views_30d, 1.0),
        avg_likes_30d=max(avg_likes_30d, 1.0),
        avg_comment_ratio_30d=max(avg_comment_ratio_30d, 0.001),
    )


def normalized_ratio(value: float, baseline: float) -> float:
    baseline = max(baseline, 1e-6)
    relative = value / baseline
    return clamp(relative / 2.0)


def harmonic_mean(values: list[float]) -> float:
    safe = [max(v, 1e-6) for v in values]
    return len(safe) / sum(1.0 / v for v in safe)


def compute_performance_score(
    view_count: int,
    like_count: int,
    comment_count: int,
    baseline: Baseline,
) -> float:
    view_norm = normalized_ratio(float(view_count), baseline.avg_views_30d)
    like_norm = normalized_ratio(float(like_count), baseline.avg_likes_30d)
    comment_ratio = (comment_count / view_count) if view_count > 0 else 0.0
    comment_norm = normalized_ratio(float(comment_ratio), baseline.avg_comment_ratio_30d)

    return clamp(harmonic_mean([view_norm, like_norm, comment_norm]))


def store_performance_snapshot(
    conn: psycopg2.extensions.connection,
    video_id: str,
    published_at: datetime,
    view_count: int,
    like_count: int,
    comment_count: int,
    score: float,
) -> None:
    now = datetime.now(timezone.utc)
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    hours_since_publish = max(int((now - published_at).total_seconds() // 3600), 0)

    like_ratio = (like_count / view_count) if view_count > 0 else None
    comment_ratio = (comment_count / view_count) if view_count > 0 else None

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO video_performance (
                video_id,
                view_count,
                like_count,
                comment_count,
                average_view_duration_seconds,
                average_view_percentage,
                retention_curve,
                like_ratio,
                comment_ratio,
                hours_since_publish
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                video_id,
                view_count,
                like_count,
                comment_count,
                None,
                None,
                None,
                like_ratio,
                comment_ratio,
                hours_since_publish,
            ),
        )

        cur.execute(
            """
            UPDATE videos
            SET performance_score = %s,
                performance_updated_at = NOW()
            WHERE id = %s
            """,
            (score, video_id),
        )


def run_default_mode(min_hours: int, dry_run: bool) -> int:
    conn = get_database_connection()
    conn.autocommit = False
    try:
        videos = fetch_videos_needing_update(conn, min_hours)
        if dry_run:
            print(f"Dry run: would process {len(videos)} videos")
            return 0

        if not videos:
            print("Processed 0 videos")
            return 0

        headers, auth_params = get_youtube_auth()
        baseline = get_latest_baseline(conn)
        processed = 0

        for batch in chunked(videos, YOUTUBE_BATCH_SIZE):
            youtube_ids = [row["youtube_id"] for row in batch if row.get("youtube_id")]
            if not youtube_ids:
                continue

            try:
                stats_map = fetch_youtube_stats_batch(youtube_ids, headers, auth_params)
            except Exception as exc:
                print(f"Batch fetch error for {len(youtube_ids)} videos: {exc}", file=sys.stderr)
                continue

            for row in batch:
                try:
                    stats = stats_map.get(row["youtube_id"])
                    if not stats:
                        print(
                            f"Missing stats for video {row['id']} (youtube_id={row['youtube_id']})",
                            file=sys.stderr,
                        )
                        continue

                    score = compute_performance_score(
                        view_count=stats["view_count"],
                        like_count=stats["like_count"],
                        comment_count=stats["comment_count"],
                        baseline=baseline,
                    )
                    store_performance_snapshot(
                        conn=conn,
                        video_id=row["id"],
                        published_at=row["published_at"],
                        view_count=stats["view_count"],
                        like_count=stats["like_count"],
                        comment_count=stats["comment_count"],
                        score=score,
                    )
                    conn.commit()
                    processed += 1
                except Exception as exc:
                    conn.rollback()
                    print(f"Failed processing video {row.get('id')}: {exc}", file=sys.stderr)

        print(f"Processed {processed} videos")
        return 0
    finally:
        conn.close()


def compute_window_baseline(
    conn: psycopg2.extensions.connection, interval_days: int
) -> tuple[float | None, float | None, float | None, int]:
    with conn.cursor() as cur:
        cur.execute(
            """
            WITH latest AS (
                SELECT DISTINCT ON (video_id)
                    video_id,
                    view_count,
                    like_count,
                    average_view_percentage,
                    fetched_at
                FROM video_performance
                ORDER BY video_id, fetched_at DESC
            )
            SELECT
                AVG(view_count)::float,
                AVG(like_count)::float,
                AVG(average_view_percentage)::float,
                COUNT(*)::int
            FROM latest
            WHERE fetched_at >= NOW() - make_interval(days => %s)
            """,
            (interval_days,),
        )
        row = cur.fetchone()

    if not row:
        return None, None, None, 0
    return row[0], row[1], row[2], row[3]


def run_update_baseline_mode(dry_run: bool) -> int:
    conn = get_database_connection()
    conn.autocommit = False
    try:
        avg_views_7d, avg_likes_7d, avg_retention_7d, count_7d = compute_window_baseline(conn, 7)
        avg_views_30d, avg_likes_30d, avg_retention_30d, count_30d = compute_window_baseline(conn, 30)

        if dry_run:
            print(
                "Dry run: baseline would be updated "
                f"(7d={count_7d} videos, 30d={count_30d} videos)"
            )
            return 0

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO channel_baselines (
                    avg_views_7d,
                    avg_likes_7d,
                    avg_retention_7d,
                    avg_views_30d,
                    avg_likes_30d,
                    avg_retention_30d,
                    video_count_7d,
                    video_count_30d
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    avg_views_7d,
                    avg_likes_7d,
                    avg_retention_7d,
                    avg_views_30d,
                    avg_likes_30d,
                    avg_retention_30d,
                    count_7d,
                    count_30d,
                ),
            )
        conn.commit()
        print("Baseline updated")
        return 0
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def run_export_mode(export_path: str, min_score: float | None) -> int:
    conn = get_database_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if min_score is not None:
                cur.execute(
                    """
                    SELECT id::text AS id, topic_brief, script, performance_score
                    FROM videos
                    WHERE status = 'published'
                      AND topic_brief IS NOT NULL
                      AND script IS NOT NULL
                      AND performance_score IS NOT NULL
                      AND performance_score >= %s
                    ORDER BY performance_score DESC
                    """,
                    (min_score,),
                )
            else:
                cur.execute(
                    """
                    SELECT id::text AS id, topic_brief, script, performance_score
                    FROM videos
                    WHERE status = 'published'
                      AND topic_brief IS NOT NULL
                      AND script IS NOT NULL
                      AND performance_score IS NOT NULL
                    ORDER BY performance_score DESC
                    """
                )
            rows = list(cur.fetchall())

        path = Path(export_path)
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        exported = 0
        with path.open("w", encoding="utf-8") as out:
            for row in rows:
                example = {
                    "video_id": row["id"],
                    "topic_brief": row["topic_brief"],
                    "script": row["script"],
                    "performance_score": float(row["performance_score"]),
                }
                out.write(json.dumps(example, ensure_ascii=False) + "\n")
                exported += 1

        print(f"Exported {exported} training examples to {export_path}")
        return 0
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Animus analytics worker: fetch stats, score videos, update baselines, export data."
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--update-baseline",
        action="store_true",
        help="Compute and insert latest channel baseline metrics",
    )
    mode_group.add_argument(
        "--export",
        type=str,
        metavar="PATH",
        help="Export JSONL training examples to PATH",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without DB/API writes",
    )
    parser.add_argument(
        "--min-hours",
        type=int,
        default=DEFAULT_MIN_HOURS,
        help=f"Minimum hours since publish in default mode (default: {DEFAULT_MIN_HOURS})",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        help="Minimum performance score filter for --export",
    )
    return parser


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()

    if args.min_hours < 0:
        print("--min-hours must be non-negative", file=sys.stderr)
        return 2
    if args.min_score is not None and not (0.0 <= args.min_score <= 1.0):
        print("--min-score must be between 0 and 1", file=sys.stderr)
        return 2

    try:
        if args.export:
            return run_export_mode(args.export, args.min_score)
        if args.update_baseline:
            return run_update_baseline_mode(args.dry_run)
        return run_default_mode(args.min_hours, args.dry_run)
    except FatalSetupError as exc:
        print(f"Fatal setup error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
