"""
Daily health check for Manuscript Search; emails the owner when something is wrong.

Runs on the pipeline desktop (WSL cron). Checks the live site and API, data
freshness per database, TLS certificates, new errors in the pipeline logs,
the nightly sync, free disk on H: and on the server, and the NAS backup.
Sends one email only when there are problems, plus a short "all good" on
Mondays so a silent checker never goes unnoticed. Email goes out through the
desktop's Outlook (COM automation, the owner's own account — no SMTP password
stored); a copy is also pushed to a private ntfy.sh topic (optional phone app).

Settings in ~/.config/mss/alerts.env (created by --setup):
    MSS_NTFY_TOPIC=<random secret topic>
    MSS_ALERT_EMAIL=<address>

    python tools/health_check.py              # check, notify if needed
    python tools/health_check.py --dry-run    # print, send nothing
    python tools/health_check.py --test       # send a test message
"""
import argparse
import json
import os
import secrets
import shutil
import socket
import ssl
import subprocess
import time
from datetime import date, datetime, timezone

import requests

SITE = "https://manuscript-search.org"
DOMAINS = ["manuscript-search.org", "msssearch.org", "zylalab.org"]
SERVER = os.environ.get("MSS_SERVER", "root@manuscript-search.org")
CONFIG_DIR = os.path.expanduser("~/.config/mss")
ENV_FILE = os.path.join(CONFIG_DIR, "alerts.env")
STATE_FILE = os.path.join(CONFIG_DIR, "health_state.json")
NAS_MARKER = r"\\100.74.173.126\homes\Dawid Zyla\manuscript_search_backup\LAST_SUCCESS"
LOCAL_DATA = "/mnt/h"

# Max age (days) of each database's last update on the live site.
MAX_AGE_DAYS = {"PubMed": 4, "BioRxiv": 4, "MedRxiv": 4, "arXiv": 10, "ClinicalTrials": 10,
                "Preprints": 10, "Grants": 10}
LOGS = ["~/pubmed.log", "~/biorxiv.log", "~/arxiv_oai.log", "~/clinicaltrials.log",
        "~/bmss_sync.log", "~/rsync.log", "~/nas_backup.log", "~/aux_index_build.log",
        "~/preprints.log", "~/grants.log"]
ERROR_PATTERNS = ("Traceback", "CRITICAL", "No space left", "Input/output error", "FAILED",
                  "kept failing", "rsync error", "Errno", "DONE with errors")


def load_env():
    env = {}
    if os.path.exists(ENV_FILE):
        for line in open(ENV_FILE):
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.strip().split("=", 1)
                env[k] = v
    return env


def setup(email: str):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    topic = f"mss-alerts-{secrets.token_urlsafe(18)}"
    with open(ENV_FILE, "w") as f:
        f.write(f"MSS_NTFY_TOPIC={topic}\nMSS_ALERT_EMAIL={email}\n")
    os.chmod(ENV_FILE, 0o600)
    print(f"Wrote {ENV_FILE}")


# ---------------------------------------------------------------------------
# Checks: each returns a list of problem strings (empty = fine)
# ---------------------------------------------------------------------------

def check_site():
    problems = []
    for path in ("/", "/health"):
        try:
            r = requests.get(SITE + path, timeout=30)
            if r.status_code != 200:
                problems.append(f"{SITE}{path} returned HTTP {r.status_code}")
        except requests.RequestException as exc:
            problems.append(f"{SITE}{path} unreachable: {exc.__class__.__name__}")
    return problems


def check_freshness():
    try:
        stats = requests.get(SITE + "/v1/stats", timeout=30).json()
    except Exception as exc:
        return [f"/v1/stats unavailable: {exc.__class__.__name__}"]
    problems, today = [], date.today()
    for name, info in stats.get("sources", {}).items():
        if info.get("available") is False:
            problems.append(f"{name} is disabled on the server: {info.get('problem')}")
        try:
            age = (today - date.fromisoformat(str(info.get("updated"))[:10])).days
        except ValueError:
            problems.append(f"{name}: no update date")
            continue
        if age > MAX_AGE_DAYS.get(name, 10):
            problems.append(f"{name} last updated {age} days ago ({info.get('updated')})")
    return problems


def check_certs():
    problems = []
    for host in DOMAINS:
        try:
            ctx = ssl.create_default_context()
            with socket.create_connection((host, 443), timeout=15) as sock:
                with ctx.wrap_socket(sock, server_hostname=host) as s:
                    not_after = s.getpeercert()["notAfter"]
            expires = datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z").replace(tzinfo=timezone.utc)
            days = (expires - datetime.now(timezone.utc)).days
            if days < 20:
                problems.append(f"TLS certificate for {host} expires in {days} days")
        except Exception as exc:
            problems.append(f"TLS check for {host} failed: {exc.__class__.__name__}")
    return problems


def check_logs(state: dict):
    problems = []
    offsets = state.setdefault("log_offsets", {})
    for log in LOGS:
        path = os.path.expanduser(log)
        if not os.path.exists(path):
            continue
        size = os.path.getsize(path)
        start = offsets.get(path, 0)
        if start > size:          # log was rotated/truncated
            start = 0
        with open(path, "rb") as f:
            f.seek(start)
            new = f.read().decode("utf-8", "replace")
        offsets[path] = size
        hits = [line.strip()[:200] for line in new.splitlines()
                if any(p in line for p in ERROR_PATTERNS)]
        if hits:
            problems.append(f"{os.path.basename(path)}: {len(hits)} error line(s), e.g. {hits[0]}")
    return problems


def check_jobs_ran():
    problems = []
    for log, max_h in (("~/bmss_sync.log", 30), ("~/pubmed.log", 30), ("~/biorxiv.log", 30)):
        path = os.path.expanduser(log)
        if not os.path.exists(path) or time.time() - os.path.getmtime(path) > max_h * 3600:
            problems.append(f"{os.path.basename(path)} not written in the last {max_h} h — did the cron job run?")
    return problems


def check_disks():
    problems = []
    free_gb = shutil.disk_usage(LOCAL_DATA).free / 1e9
    if free_gb < 25:
        problems.append(f"H: drive has only {free_gb:.0f} GB free")
    try:
        out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", SERVER,
                              "df -BG --output=avail / | tail -1; free -m | awk '/Mem:/{print $7}'"],
                             capture_output=True, text=True, timeout=60).stdout.split()
        if int(out[0].rstrip("G")) < 15:
            problems.append(f"server disk has only {out[0]} free")
        if int(out[1]) < 500:
            problems.append(f"server has only {out[1]} MB memory available")
    except Exception as exc:
        problems.append(f"could not check the server over SSH: {exc.__class__.__name__}")
    return problems


def check_nas_backup():
    try:
        out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", f"Get-Content '{NAS_MARKER}'"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        when = datetime.fromisoformat(out.splitlines()[0].strip())
        hours = (datetime.now(when.tzinfo) - when).total_seconds() / 3600
        return [f"last successful NAS backup was {hours:.0f} h ago"] if hours > 30 else []
    except Exception as exc:
        return [f"NAS backup marker unreadable ({exc.__class__.__name__}) — has the backup ever succeeded?"]


def weekly_usage() -> str:
    """Plain-text usage summary for the last 7 days (counts and timings only)."""
    try:
        out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", SERVER,
                              "cd /root/pubmed_search && /root/space_env/bin/python tools/usage_report.py --days 7"],
                             capture_output=True, text=True, timeout=180).stdout
        u = json.loads(out.strip().splitlines()[-1])
    except Exception as exc:
        return f"Usage report unavailable ({exc.__class__.__name__})."
    days = u.get("per_day", {})
    searches = sum(d["searches"] for d in days.values())
    similar = sum(d["similar"] for d in days.values())
    lat = u.get("latency", {})
    lines = [f"Last 7 days: {searches:,} searches and {similar:,} 'similar papers' requests"
             + (f"; median {lat['p50_s']} s, 95% under {lat['p95_s']} s, slowest {lat['max_s']} s." if lat.get("p50_s") else ".")]
    busiest = max(days.items(), key=lambda kv: kv[1]["searches"] + kv[1]["similar"], default=None)
    if busiest:
        lines.append(f"Busiest day: {busiest[0]} ({busiest[1]['searches'] + busiest[1]['similar']:,}).")
    tiers = u.get("api_by_tier", {})
    if "error" in tiers:
        lines.append(f"API usage unavailable: {tiers['error']}")
    elif tiers:
        lines.append("API and MCP: " + "; ".join(
            f"{t} {v['requests']:,} request{'s' if v['requests'] != 1 else ''} from {v['callers']:,} "
            f"{'IP' if t == 'anonymous' else 'key'}{'s' if v['callers'] != 1 else ''}"
            for t, v in sorted(tiers.items())) + ".")
    else:
        lines.append("No API or MCP use.")
    if u.get("new_keys"):
        lines.append(f"New API keys: {u['new_keys']}.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------

def send_outlook(to: str, subject: str, body: str):
    """Sends through the desktop's Outlook via COM (no credentials stored)."""
    import base64
    b64 = lambda t: base64.b64encode(t.encode("utf-8")).decode()   # noqa: E731
    script = (
        "$d = { param($x) [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($x)) };"
        "$ol = New-Object -ComObject Outlook.Application;"
        "$m = $ol.CreateItem(0);"
        f"$m.To = (& $d '{b64(to)}'); $m.Subject = (& $d '{b64(subject)}'); $m.Body = (& $d '{b64(body)}');"
        "$m.Send(); try { $ol.Session.SendAndReceive($false) } catch {}"
    )
    r = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                       capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f"Outlook send failed: {r.stderr.strip()[:300]}")


def send(env: dict, title: str, body: str, priority: str = "default"):
    topic, email = env.get("MSS_NTFY_TOPIC"), env.get("MSS_ALERT_EMAIL")
    if not email and not topic:
        raise SystemExit(f"No MSS_ALERT_EMAIL/MSS_NTFY_TOPIC in {ENV_FILE}; run with --setup EMAIL first.")
    errors = []
    if email:
        try:
            send_outlook(email, title, body + f"\n\n— daily check on {socket.gethostname()}, {datetime.now():%Y-%m-%d %H:%M}")
        except Exception as exc:
            errors.append(str(exc))
    if topic:
        try:
            requests.post(f"https://ntfy.sh/{topic}", data=body.encode("utf-8"), timeout=30,
                          headers={"Title": title, "Priority": priority, "Tags": "microscope"}).raise_for_status()
        except Exception as exc:
            errors.append(f"ntfy: {exc}")
    # Email is the channel that matters; the ntfy push is best effort.
    if email and any(e.startswith("Outlook") for e in errors):
        raise RuntimeError("; ".join(errors))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--setup", metavar="EMAIL")
    args = ap.parse_args()

    if args.setup:
        setup(args.setup)
        return
    env = load_env()
    if args.test:
        send(env, "Manuscript Search alerts are set up",
             "This is a test. You will get an email like this when the daily check finds a problem, "
             "and a short 'all good' every Monday.")
        print("test message sent")
        return

    state = json.load(open(STATE_FILE)) if os.path.exists(STATE_FILE) else {}
    sections = {
        "Site": check_site(), "Data freshness": check_freshness(), "Certificates": check_certs(),
        "Pipeline logs": check_logs(state), "Scheduled jobs": check_jobs_ran(),
        "Disk": check_disks(), "NAS backup": check_nas_backup(),
    }
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

    problems = [(s, p) for s, ps in sections.items() for p in ps]
    monday = date.today().weekday() == 0
    if problems:
        title = f"Manuscript Search: {len(problems)} problem{'s' if len(problems) > 1 else ''}"
        body = "\n".join(f"• [{s}] {p}" for s, p in problems)
        priority = "high"
    elif monday:
        title, body, priority = "Manuscript Search: all good", "Weekly check-in: every daily check passed.", "low"
    else:
        print("all checks passed")
        return
    if monday:
        body += "\n\nUsage\n" + weekly_usage()
    print(title + "\n" + body)
    if not args.dry_run:
        send(env, title, body, priority)


if __name__ == "__main__":
    main()
