"""
API access: keys, tiers and daily quotas (SQLite; keys stored only as SHA-256).

Tiers (searches per UTC day):
    anonymous   no key, counted per client IP        ANON_DAILY_LIMIT (20)
    free        self-service key from /signup        FREE_DAILY_LIMIT (1,000)
    partner     keys issued by the owner             PARTNER_DAILY_LIMIT (10,000)
    internal    the web UI (MSS_INTERNAL_KEY)        unlimited

Admin CLI (run on the server, from /root/pubmed_search):
    python access.py create --tier partner --label "Lab X" [--email a@b]
    python access.py revoke <key-or-prefix>
    python access.py list
    python access.py usage [--days 7]
"""
import argparse
import hashlib
import os
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timezone

DB_PATH = os.environ.get("MSS_ACCESS_DB",
                         os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "access.sqlite3"))
LIMITS = {
    "anonymous": int(os.environ.get("MSS_ANON_DAILY_LIMIT", "20")),
    "free": int(os.environ.get("MSS_FREE_DAILY_LIMIT", "1000")),
    "partner": int(os.environ.get("MSS_PARTNER_DAILY_LIMIT", "10000")),
}
SIGNUPS_PER_IP_PER_DAY = 3
_lock = threading.Lock()


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS keys (
            key_hash TEXT PRIMARY KEY, prefix TEXT, tier TEXT NOT NULL, label TEXT, email TEXT,
            created INTEGER NOT NULL, revoked INTEGER NOT NULL DEFAULT 0, daily_limit INTEGER);
        CREATE TABLE IF NOT EXISTS usage (
            subject TEXT NOT NULL, day TEXT NOT NULL, count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (subject, day));
        CREATE TABLE IF NOT EXISTS signups (ip TEXT, email TEXT, created INTEGER);
    """)
    return conn


def create_key(tier: str, label: str = "", email: str = "", key: str = None) -> str:
    """Creates (or imports) a key and returns it. The plain key is not stored."""
    if tier not in LIMITS:
        raise ValueError(f"tier must be one of {sorted(LIMITS)}")
    key = key or f"mss_{secrets.token_urlsafe(24)}"
    with _lock, _connect() as conn:
        conn.execute("INSERT OR IGNORE INTO keys (key_hash, prefix, tier, label, email, created) VALUES (?,?,?,?,?,?)",
                     (_hash(key), key[:10], tier, label, email, int(time.time())))
    return key


def import_legacy_keys(path: str) -> int:
    """Imports api_keys.txt (one key per line) as partner keys; returns how many were new."""
    if not os.path.exists(path):
        return 0
    new = 0
    with _connect() as conn:
        known = {r[0] for r in conn.execute("SELECT key_hash FROM keys")}
    for line in open(path):
        key = line.strip()
        if key and not key.startswith("#") and _hash(key) not in known:
            create_key("partner", label="imported from api_keys.txt", key=key)
            new += 1
    return new


def lookup(key: str):
    """(tier, subject, daily_limit) for an active key, else None."""
    with _connect() as conn:
        row = conn.execute("SELECT tier, daily_limit FROM keys WHERE key_hash=? AND revoked=0",
                           (_hash(key),)).fetchone()
    if not row:
        return None
    tier, limit = row
    return tier, f"key:{_hash(key)[:16]}", limit or LIMITS[tier]


def consume(subject: str, limit: int):
    """Counts one request against today's quota. Returns (allowed, remaining)."""
    day = _today()
    with _lock, _connect() as conn:
        row = conn.execute("SELECT count FROM usage WHERE subject=? AND day=?", (subject, day)).fetchone()
        used = row[0] if row else 0
        if used >= limit:
            return False, 0
        conn.execute("INSERT INTO usage (subject, day, count) VALUES (?,?,1) "
                     "ON CONFLICT(subject, day) DO UPDATE SET count = count + 1", (subject, day))
        # keep 60 days of history
        conn.execute("DELETE FROM usage WHERE day < date('now', '-60 day')")
    return True, limit - used - 1


def signup_allowed(ip: str, email: str):
    """None if allowed, else a reason."""
    since = int(time.time()) - 86400
    with _connect() as conn:
        n_ip = conn.execute("SELECT COUNT(*) FROM signups WHERE ip=? AND created>?", (ip, since)).fetchone()[0]
        n_mail = conn.execute("SELECT COUNT(*) FROM signups WHERE lower(email)=lower(?) AND created>?",
                              (email, since)).fetchone()[0]
    if n_ip >= SIGNUPS_PER_IP_PER_DAY:
        return "Too many signups from this network today; try again tomorrow."
    if n_mail:
        return "A key was already sent to this address today; check your inbox (and spam folder)."
    return None


def record_signup(ip: str, email: str):
    with _lock, _connect() as conn:
        conn.execute("INSERT INTO signups (ip, email, created) VALUES (?,?,?)", (ip, email, int(time.time())))
        # one active free key per address: revoke older ones
        conn.execute("UPDATE keys SET revoked=1 WHERE tier='free' AND lower(email)=lower(?)", (email,))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--tier", default="partner", choices=sorted(LIMITS))
    c.add_argument("--label", default="")
    c.add_argument("--email", default="")
    r = sub.add_parser("revoke")
    r.add_argument("key_or_prefix")
    sub.add_parser("list")
    u = sub.add_parser("usage")
    u.add_argument("--days", type=int, default=7)
    args = ap.parse_args()

    if args.cmd == "create":
        print(create_key(args.tier, args.label, args.email))
    elif args.cmd == "revoke":
        k = args.key_or_prefix
        with _connect() as conn:
            n = conn.execute("UPDATE keys SET revoked=1 WHERE key_hash=? OR prefix=?", (_hash(k), k[:10])).rowcount
        print(f"revoked {n} key(s)")
    elif args.cmd == "list":
        with _connect() as conn:
            for row in conn.execute("SELECT prefix, tier, label, email, datetime(created,'unixepoch'), revoked "
                                    "FROM keys ORDER BY created"):
                print(" | ".join(str(x) for x in row))
    elif args.cmd == "usage":
        with _connect() as conn:
            for row in conn.execute("SELECT day, subject, count FROM usage WHERE day >= date('now', ?) "
                                    "ORDER BY day DESC, count DESC", (f"-{args.days} day",)):
                print(" | ".join(str(x) for x in row))


if __name__ == "__main__":
    main()
