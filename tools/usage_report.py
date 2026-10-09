"""
Weekly usage summary, run on the server (by tools/health_check.py over SSH).
No query text is logged or reported — only counts and timings.

    python tools/usage_report.py [--days 7]     # prints JSON
"""
import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LINE = re.compile(r"^(\d{4}-\d{2}-\d{2}) \S+ \[INFO\] search_api — (Search|Similar) .* in ([\d.]+)s")


def search_log(days: int):
    """Per-day counts and latencies from the backend's own log lines."""
    out = subprocess.run(["journalctl", "-u", "mss-backend", "--since", f"-{days} days", "-o", "cat",
                          "--no-pager"], capture_output=True, text=True, timeout=120).stdout
    per_day, times = defaultdict(lambda: {"searches": 0, "similar": 0}), []
    for line in out.splitlines():
        m = LINE.match(line)
        if not m:
            continue
        day, kind, secs = m.groups()
        per_day[day]["similar" if kind == "Similar" else "searches"] += 1
        times.append(float(secs))
    times.sort()
    pct = lambda p: round(times[min(len(times) - 1, int(p * len(times)))], 2) if times else None   # noqa: E731
    return dict(sorted(per_day.items())), {"p50_s": pct(0.5), "p95_s": pct(0.95), "max_s": times[-1] if times else None}


def api_usage(days: int):
    import access
    with sqlite3.connect(access.DB_PATH) as conn:
        rows = conn.execute("SELECT subject, SUM(count) FROM usage WHERE day >= date('now', ?) GROUP BY subject",
                            (f"-{days} day",)).fetchall()
        tiers = dict(conn.execute("SELECT 'key:' || substr(key_hash, 1, 16), tier FROM keys").fetchall())
        new_keys = conn.execute("SELECT COUNT(*) FROM keys WHERE created >= strftime('%s', 'now', ?)",
                                (f"-{days} day",)).fetchone()[0]
    by_tier = defaultdict(lambda: {"requests": 0, "callers": 0})
    for subject, n in rows:
        tier = "anonymous" if subject.startswith("ip:") else tiers.get(subject, "unknown")
        by_tier[tier]["requests"] += n
        by_tier[tier]["callers"] += 1
    return dict(by_tier), new_keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()
    per_day, latency = search_log(args.days)
    try:
        tiers, new_keys = api_usage(args.days)
    except Exception as exc:
        tiers, new_keys = {"error": str(exc)}, None
    print(json.dumps({"days": args.days, "per_day": per_day, "latency": latency,
                      "api_by_tier": tiers, "new_keys": new_keys}))


if __name__ == "__main__":
    main()
