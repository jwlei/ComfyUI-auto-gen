#!/usr/bin/env python3
"""
Uploads finished films from output/ to YouTube and moves them to uploaded/.

The LLM that planned each film writes its title, description and tags. The web
UI's Upload and Upload all buttons run this.

    python publish.py The_Storm_run_20261101_203000.mp4 --privacy unlisted
    python publish.py --all
"""
import argparse
import os
import sys

from dotenv import load_dotenv

from modules import paths

load_dotenv(os.path.join(paths.PROJECT_ROOT, ".env"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Upload finished films from output/ to YouTube.")
    parser.add_argument("videos", nargs="*", help="File names in output/.")
    parser.add_argument("--all", action="store_true", help="Upload every film in output/.")
    parser.add_argument("--privacy", choices=("private", "unlisted", "public"), default="private",
                        help="Privacy status (default: private).")
    args = parser.parse_args(argv)
    if args.all == bool(args.videos):
        parser.error("name the films to upload, or pass --all")
    return args


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    if args.all:
        names = sorted(n for n in os.listdir(paths.OUTPUT_DIR) if n.lower().endswith(".mp4")) if os.path.isdir(paths.OUTPUT_DIR) else []
        if not names:
            print("✅ Nothing in output/ to upload.")
            return 0
    else:
        names = [os.path.basename(v) for v in args.videos]

    try:
        # Imported here so a missing Google client library is reported, not a crash.
        from modules.publisher import publish_film
    except ImportError as e:
        print(f"❌ Upload Error: {e}. Run: pip install -r requirements.txt")
        return 1

    uploaded, failed = 0, []
    for number, name in enumerate(names, 1):
        print(f"\n📼 {number}/{len(names)}: {name}")
        path = os.path.join(paths.OUTPUT_DIR, name)
        url = None
        if not os.path.exists(path):
            print(f"❌ {name} is not in output/.")
        else:
            try:
                url = publish_film(path, args.privacy)
            except Exception as e:
                print(f"❌ Upload Error: {e}")
        if url:
            uploaded += 1
        else:
            failed.append(name)

    print(f"\n📦 Uploaded {uploaded}/{len(names)} film(s).")
    for name in failed:
        print(f"   • not uploaded: {name}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
