#!/usr/bin/env python3
"""
Upload agent traces to HuggingFace Hub — Sharing is Caring bonus quest.

Usage:
    python scripts/upload_traces.py [--trace-dir /tmp/tiny-therapist-traces] [--repo-id leomcamilo/tiny-therapist-traces]

Requires: huggingface_hub (pip install huggingface_hub)
Auth: huggingface-cli login
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_traces(trace_dir: str) -> list[dict]:
    """Load all traces from the JSONL file."""
    trace_file = os.path.join(trace_dir, "traces.jsonl")
    if not os.path.exists(trace_file):
        print(f"No traces found at {trace_file}")
        return []

    traces = []
    with open(trace_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    return traces


def upload_traces(trace_dir: str, repo_id: str, dry_run: bool = False):
    """Upload traces to HuggingFace Hub as a dataset."""
    traces = load_traces(trace_dir)
    if not traces:
        print("No traces to upload.")
        return

    print(f"Found {len(traces)} traces")

    if dry_run:
        print(f"[DRY RUN] Would upload {len(traces)} traces to {repo_id}")
        for i, t in enumerate(traces[:3]):
            print(f"  [{i+1}] mode={t['mode']} backend={t['backend']}")
        if len(traces) > 3:
            print(f"  ... and {len(traces)-3} more")
        return

    from huggingface_hub import HfApi

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"Error creating repo: {e}")
        return

    # Write traces to a temp file and upload
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
        temp_path = f.name

    try:
        api.upload_file(
            path_or_fileobj=temp_path,
            path_in_repo="traces.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"✅ Uploaded {len(traces)} traces to https://huggingface.co/datasets/{repo_id}")
    finally:
        os.unlink(temp_path)


def main():
    parser = argparse.ArgumentParser(description="Upload agent traces to HuggingFace Hub")
    parser.add_argument(
        "--trace-dir",
        default=os.environ.get("TRACE_DIR", "/tmp/tiny-therapist-traces"),
        help="Directory containing traces.jsonl",
    )
    parser.add_argument(
        "--repo-id",
        default="leomcamilo/tiny-therapist-traces",
        help="HuggingFace dataset repo ID",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without uploading")
    args = parser.parse_args()

    upload_traces(args.trace_dir, args.repo_id, args.dry_run)


if __name__ == "__main__":
    main()