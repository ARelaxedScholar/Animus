#!/usr/bin/env python3
"""
Animus Farm Remote Control

Commands:
  status              - Get current farm status
  upload <file>       - Upload a manual script
  analytics           - Fetch YouTube analytics for published videos
  analytics --export  - Export training data for DSPy
  analytics --baseline - Update channel baseline metrics
"""
import requests
import subprocess
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

def run_analytics(args):
    """Run the analytics worker with the given arguments."""
    script_path = os.path.join(os.path.dirname(__file__), "scripts", "analytics_worker.py")
    
    if not os.path.exists(script_path):
        print(f"Error: Analytics worker not found at {script_path}")
        return
    
    cmd = [sys.executable, script_path]
    
    if args.export:
        cmd.extend(["--export", args.export])
        if args.min_score:
            cmd.extend(["--min-score", str(args.min_score)])
    elif args.baseline:
        cmd.append("--update-baseline")
    else:
        # Default: fetch analytics for videos needing updates
        if args.dry_run:
            cmd.append("--dry-run")
        if args.min_hours:
            cmd.extend(["--min-hours", str(args.min_hours)])
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

def show_training_stats():
    """Show statistics about available training data."""
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            print("Error: DATABASE_URL not set")
            return
        
        conn = psycopg2.connect(database_url)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Count videos with performance scores
            cur.execute("""
                SELECT 
                    COUNT(*) FILTER (WHERE performance_score IS NOT NULL) as scored,
                    COUNT(*) FILTER (WHERE performance_score IS NULL AND status = 'published') as unscored,
                    AVG(performance_score) as avg_score,
                    MIN(performance_score) as min_score,
                    MAX(performance_score) as max_score
                FROM videos
                WHERE status = 'published'
            """)
            stats = cur.fetchone()
            
            # Get score distribution
            cur.execute("""
                SELECT 
                    CASE 
                        WHEN performance_score >= 0.8 THEN 'excellent (0.8+)'
                        WHEN performance_score >= 0.6 THEN 'good (0.6-0.8)'
                        WHEN performance_score >= 0.4 THEN 'average (0.4-0.6)'
                        ELSE 'below average (<0.4)'
                    END as tier,
                    COUNT(*) as count
                FROM videos
                WHERE performance_score IS NOT NULL
                GROUP BY tier
                ORDER BY tier
            """)
            distribution = cur.fetchall()
        
        conn.close()
        
        print("\n--- Training Data Statistics ---")
        print(f"Videos with scores: {stats['scored']}")
        print(f"Videos pending scores: {stats['unscored']}")
        
        if stats['avg_score']:
            print(f"\nScore range: {stats['min_score']:.3f} - {stats['max_score']:.3f}")
            print(f"Average score: {stats['avg_score']:.3f}")
        
        if distribution:
            print("\nScore distribution:")
            for row in distribution:
                print(f"  {row['tier']}: {row['count']} videos")
        
    except ImportError:
        print("Error: psycopg2 not installed. Run: pip install psycopg2-binary")
    except Exception as e:
        print(f"Error: {e}")

def compile_judge(args):
    """Compile/optimize the DSPy Judge from training data."""
    import tempfile
    
    bridge_path = os.path.join(os.path.dirname(__file__), "src", "bridge", "dspy_bridge.py")
    
    if not os.path.exists(bridge_path):
        print(f"Error: DSPy bridge not found at {bridge_path}")
        return
    
    training_data_path = args.training_data
    
    # If no training data provided, export from database first
    if not training_data_path:
        print("No training data provided. Exporting from database...")
        
        # Create a temp file for training data
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False)
        training_data_path = temp_file.name
        temp_file.close()
        
        # Export training data via analytics worker
        analytics_script = os.path.join(os.path.dirname(__file__), "scripts", "analytics_worker.py")
        export_cmd = [sys.executable, analytics_script, "--export", training_data_path]
        
        result = subprocess.run(export_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Export failed: {result.stderr}")
            os.unlink(training_data_path)
            return
        
        print(f"Exported training data to {training_data_path}")
    
    # Check minimum examples
    example_count = 0
    with open(training_data_path, 'r') as f:
        for line in f:
            if line.strip():
                example_count += 1
    
    if example_count < args.min_examples:
        print(f"Error: Only {example_count} training examples available.")
        print(f"       Minimum required: {args.min_examples}")
        print("       Publish more videos and collect analytics before compiling.")
        if not args.training_data:
            os.unlink(training_data_path)
        return
    
    print(f"Compiling Judge with {example_count} training examples...")
    
    # Run the DSPy bridge in compile mode
    cmd = [sys.executable, bridge_path, "--compile", "--training-data", training_data_path]
    
    if args.output:
        cmd.extend(["--output", args.output])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print stderr (DSPy logs progress there)
    if result.stderr:
        for line in result.stderr.strip().split('\n'):
            print(f"  {line}")
    
    if result.returncode != 0:
        print(f"\nCompilation failed!")
        return
    
    # Parse the result
    try:
        output = json.loads(result.stdout)
        if output.get("success"):
            print(f"\nJudge compiled successfully!")
            print(f"Saved to: {output.get('compiled_path')}")
            print("\nThe DSPy Judge will now use learned patterns from real-world")
            print("video performance to predict script quality.")
        else:
            print(f"\nCompilation failed: {output.get('error')}")
    except json.JSONDecodeError:
        print(f"Unexpected output: {result.stdout}")
    
    # Cleanup temp file if we created one
    if not args.training_data and os.path.exists(training_data_path):
        os.unlink(training_data_path)

def main():
    parser = argparse.ArgumentParser(
        description="Animus Farm Remote Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  farm_ctl.py status                    # Check farm status
  farm_ctl.py analytics                 # Fetch analytics for published videos
  farm_ctl.py analytics --dry-run       # Preview what would be fetched
  farm_ctl.py analytics --baseline      # Update channel baseline
  farm_ctl.py analytics --export data.jsonl   # Export training data
  farm_ctl.py training-stats            # Show training data statistics
  farm_ctl.py compile-judge             # Compile/optimize the DSPy Judge
        """
    )
    subparsers = parser.add_subparsers(dest="command")

    # Status command
    subparsers.add_parser("status", help="Get current farm status")
    
    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload a manual script")
    upload_parser.add_argument("file", help="Path to JSON script file")
    
    # Analytics command
    analytics_parser = subparsers.add_parser("analytics", 
        help="Fetch YouTube analytics and compute performance scores")
    analytics_parser.add_argument("--export", type=str, metavar="PATH",
        help="Export training data to JSONL file instead of fetching")
    analytics_parser.add_argument("--baseline", action="store_true",
        help="Update channel baseline metrics only")
    analytics_parser.add_argument("--dry-run", action="store_true",
        help="Show what would be processed without making changes")
    analytics_parser.add_argument("--min-hours", type=int, default=168,
        help="Minimum hours since publish (default: 168 = 7 days)")
    analytics_parser.add_argument("--min-score", type=float,
        help="Minimum score for export (0.0-1.0)")
    
    # Training stats command
    subparsers.add_parser("training-stats", 
        help="Show statistics about available training data")
    
    # Compile Judge command
    compile_parser = subparsers.add_parser("compile-judge",
        help="Compile/optimize the DSPy Judge from training data")
    compile_parser.add_argument("--training-data", type=str, metavar="PATH",
        help="Path to training data JSONL (default: auto-export from database)")
    compile_parser.add_argument("--output", type=str, metavar="PATH",
        help="Output path for compiled program")
    compile_parser.add_argument("--min-examples", type=int, default=10,
        help="Minimum training examples required (default: 10)")

    args = parser.parse_args()
    
    if args.command == "status":
        get_status()
    elif args.command == "upload":
        upload_script(args.file)
    elif args.command == "analytics":
        run_analytics(args)
    elif args.command == "training-stats":
        show_training_stats()
    elif args.command == "compile-judge":
        compile_judge(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
