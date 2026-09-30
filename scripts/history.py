#!/usr/bin/env python3
"""Maintain dte/history/YYYY-MM.csv: one row per DTE outage job, built from snapshots.

Each snapshot of dte/outages.geojson is a list of *currently active* outages.
Folding successive snapshots together gives each job a first/last-seen time,
which is what the dashboard needs for history, durations and trends.
Rows are partitioned by the month the job was first seen, so each run only
rewrites the current month's file and git history stays small.

Usage:
    # fold the current snapshot into the history (used by the workflow)
    python3 scripts/history.py update

    # rebuild the history from every snapshot in git (one-off, ~1-2 min)
    python3 scripts/history.py backfill

Standard library only, so it runs on a bare GitHub Actions runner.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SNAPSHOT = Path("dte/outages.geojson")
HISTORY = Path("dte/history")

FIELDS = [
    "job_id", "first_seen", "last_seen", "snapshots",
    "off_time", "est_restore", "customers_max", "customers_last",
    "cause", "event_status", "incident_type", "device_type",
    "service_center", "storm_mode", "lat", "lon",
]

# snapshot property -> history column; the latest non-null value wins
LATEST = {
    "CAUSE": "cause",
    "EVENT_STATUS": "event_status",
    "TYCOD": "incident_type",
    "DEV_TYPE_NAME": "device_type",
    "SERVICE_CENTER": "service_center",
    "STORM_MODE": "storm_mode",
}


def iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def epoch_ms_to_iso(value) -> str:
    if value in (None, ""):
        return ""
    return iso(datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc))


def centroid(geometry: dict | None) -> tuple[float, float] | None:
    """Area-weighted centroid of the largest outer ring, as (lat, lon)."""
    if not geometry or not geometry.get("coordinates"):
        return None
    coords = geometry["coordinates"]
    rings = [coords[0]] if geometry["type"] == "Polygon" else [p[0] for p in coords]
    best = None
    for ring in rings:
        a = cx = cy = 0.0
        for (x0, y0), (x1, y1) in zip(ring, ring[1:]):
            cross = x0 * y1 - x1 * y0
            a += cross
            cx += (x0 + x1) * cross
            cy += (y0 + y1) * cross
        if a:
            cand = (abs(a), cy / (3 * a), cx / (3 * a))
        else:  # degenerate ring: fall back to the vertex mean
            cand = (0.0, sum(p[1] for p in ring) / len(ring), sum(p[0] for p in ring) / len(ring))
        if best is None or cand[0] > best[0]:
            best = cand
    return round(best[1], 5), round(best[2], 5)


def load_history(directory: Path) -> dict[str, dict]:
    rows = {}
    for path in sorted(directory.glob("*.csv")):
        with path.open(newline="") as f:
            rows.update((row["job_id"], row) for row in csv.DictReader(f))
    return rows


def save_history(rows: dict[str, dict], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    months: dict[str, list[dict]] = {}
    for row in rows.values():
        months.setdefault(row["first_seen"][:7], []).append(row)
    for month, month_rows in months.items():
        path = directory / f"{month}.csv"
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(sorted(month_rows, key=lambda r: (r["first_seen"], r["job_id"])))
        tmp.replace(path)


def fold(rows: dict[str, dict], snapshot: dict, seen_at: str) -> int:
    """Merge one snapshot into rows. Returns the number of jobs touched."""
    touched = 0
    for feature in snapshot.get("features", []):
        props = feature.get("properties") or {}
        job = props.get("JOB_ID")
        # DTE leaves some stale polygons with no outage data (e.g. a 2024 job
        # present in every snapshot for a year); they are not real outages.
        if not job or (props.get("NUM_CUST") is None and props.get("OFF_DTTM") is None):
            continue
        row = rows.get(job)
        if row is None:
            row = rows[job] = {k: "" for k in FIELDS}
            row.update(job_id=job, first_seen=seen_at, snapshots="0")
            where = centroid(feature.get("geometry"))
            if where:
                row["lat"], row["lon"] = where
        elif row["last_seen"] and seen_at <= row["last_seen"]:
            continue  # already folded this snapshot (idempotent re-runs)

        row["last_seen"] = seen_at
        row["snapshots"] = str(int(row["snapshots"] or 0) + 1)
        for prop, col in LATEST.items():
            if props.get(prop) not in (None, ""):
                row[col] = str(props[prop]).strip()
        if props.get("OFF_DTTM"):
            row["off_time"] = epoch_ms_to_iso(props["OFF_DTTM"])
        if props.get("EST_REP_DTTM"):
            row["est_restore"] = epoch_ms_to_iso(props["EST_REP_DTTM"])
        customers = props.get("NUM_CUST")
        if customers is not None:
            row["customers_last"] = str(customers)
            row["customers_max"] = str(max(int(customers), int(row["customers_max"] or 0)))
        touched += 1
    return touched


def cmd_update(args) -> None:
    rows = load_history(args.history)
    snapshot = json.loads(args.snapshot.read_text())
    seen_at = args.seen_at or iso(datetime.now(timezone.utc))
    n = fold(rows, snapshot, seen_at)
    save_history(rows, args.history)
    print(f"history: folded {n} active jobs at {seen_at}; {len(rows)} jobs total")


def iter_snapshots(snapshot_path: Path = SNAPSHOT):
    """Yield (seen_at_iso, snapshot_dict) for every committed version of the snapshot, oldest first."""
    log = subprocess.run(
        ["git", "log", "--reverse", "--format=%H %cI", "--", str(snapshot_path)],
        check=True, capture_output=True, text=True,
    ).stdout.split()
    commits = list(zip(log[::2], log[1::2]))
    print(f"replaying {len(commits)} snapshots from git history", file=sys.stderr)
    batch = subprocess.Popen(["git", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    try:
        for i, (sha, when) in enumerate(commits, 1):
            batch.stdin.write(f"{sha}:{snapshot_path.as_posix()}\n".encode())
            batch.stdin.flush()
            header = batch.stdout.readline().split()
            if header[-1] == b"missing":
                continue
            body = batch.stdout.read(int(header[2]))
            batch.stdout.read(1)  # trailing newline
            if i % 1000 == 0:
                print(f"  {i}/{len(commits)} snapshots", file=sys.stderr)
            try:
                yield iso(datetime.fromisoformat(when)), json.loads(body)
            except json.JSONDecodeError:
                continue
    finally:
        batch.stdin.close()
        batch.wait()


def cmd_backfill(args) -> None:
    rows: dict[str, dict] = {}
    for seen_at, snapshot in iter_snapshots(args.snapshot):
        fold(rows, snapshot, seen_at)
    save_history(rows, args.history)
    print(f"backfill: wrote {len(rows)} jobs to {args.history}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    parser.add_argument("--history", type=Path, default=HISTORY)
    sub = parser.add_subparsers(dest="cmd", required=True)
    update = sub.add_parser("update", help="fold the current snapshot into the history")
    update.add_argument("--seen-at", help="ISO-8601 UTC time of the snapshot (default: now)")
    update.set_defaults(func=cmd_update)
    sub.add_parser("backfill", help="rebuild the history from git").set_defaults(func=cmd_backfill)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
