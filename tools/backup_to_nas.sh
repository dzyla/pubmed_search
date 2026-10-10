#!/usr/bin/env bash
# Nightly backup to the lab NAS (run from WSL cron on the pipeline desktop).
#
#  1. Pipeline data (snowflake/: embeddings + parquet, the source of truth the
#     server is synced from) → NAS, incremental via robocopy. Never deletes on
#     the NAS (no /MIR), so an accidental local deletion does not propagate.
#  2. Server configuration + secrets (nginx, systemd units, certificates, .env,
#     api_keys.txt, ssh hardening, RustDesk keys) → dated tarball, keep 8.
#  3. Writes LAST_SUCCESS (read by tools/health_check.py).
#
# Settings (env or defaults below): MSS_NAS_UNC, MSS_DATA_WIN, MSS_SERVER.
set -uo pipefail

NAS_UNC="${MSS_NAS_UNC:-\\\\100.74.173.126\\homes\\Dawid Zyla\\manuscript_search_backup}"
DATA_WIN="${MSS_DATA_WIN:-H:\\pubmed_semantic_search\\pubmed_semantic_search\\snowflake}"
SERVER="${MSS_SERVER:-root@manuscript-search.org}"
LOG="${HOME}/nas_backup.log"
stamp=$(date +%Y%m%d-%H%M%S)
ok=1

# cron's PATH has no Windows folders: call the Windows tools by full path
PWSH=$(command -v powershell.exe || echo /mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe)
ROBOCOPY=$(command -v robocopy.exe || echo /mnt/c/Windows/System32/Robocopy.exe)
pwsh_run() { "$PWSH" -NoProfile -NonInteractive -Command "$1" 2>&1 | tr -d '\r'; }

echo "[$(date -Is)] START backup → $NAS_UNC" >> "$LOG"

# 1. pipeline data. robocopy exit codes 0-7 = success, >= 8 = failure.
cd /mnt/c || exit 1
"$ROBOCOPY" "$DATA_WIN" "$NAS_UNC\\snowflake" /E /COPY:DAT /DCOPY:T /R:2 /W:30 /MT:8 \
    /XF "*.tmp" "*.tmp.npy" /XD gpu_locks /NP /NFL /NDL /NJH >> "$LOG" 2>&1
rc=$?
if [ "$rc" -ge 8 ]; then
    echo "[$(date -Is)] robocopy FAILED (exit $rc)" >> "$LOG"; ok=0
fi

# 2. server configuration snapshot (secrets included; the NAS home share is private)
tmp=$(mktemp -d)
if ssh -o BatchMode=yes -o ConnectTimeout=20 "$SERVER" \
     'tar czf - --ignore-failed-read /etc/nginx /etc/systemd/system/mss-*.service /etc/letsencrypt \
        /etc/ssh/sshd_config.d /etc/iptables /root/pubmed_search/.env /root/pubmed_search/api_keys.txt \
        /root/pubmed_search/config_mss.yaml /var/lib/rustdesk-server /var/spool/cron/crontabs 2>/dev/null' \
     > "$tmp/server-config-$stamp.tgz" && [ -s "$tmp/server-config-$stamp.tgz" ]; then
    pwsh_run "New-Item -ItemType Directory -Force -Path '$NAS_UNC\\server' | Out-Null;
        Copy-Item -Path '$(wslpath -w "$tmp/server-config-$stamp.tgz")' -Destination '$NAS_UNC\\server\\';
        Get-ChildItem '$NAS_UNC\\server\\server-config-*.tgz' | Sort-Object Name -Descending |
          Select-Object -Skip 8 | Remove-Item" >> "$LOG"
else
    echo "[$(date -Is)] server config snapshot FAILED" >> "$LOG"; ok=0
fi
crontab -l > "$tmp/crontab-desktop.txt" 2>/dev/null && \
    pwsh_run "Copy-Item -Path '$(wslpath -w "$tmp/crontab-desktop.txt")' -Destination '$NAS_UNC\\' -Force" >> "$LOG"
rm -rf "$tmp"

# 3. success marker
if [ "$ok" -eq 1 ]; then
    pwsh_run "Set-Content -Path '$NAS_UNC\\LAST_SUCCESS' -Value '$(date -Is)'" >> "$LOG"
    echo "[$(date -Is)] DONE ok" >> "$LOG"
else
    echo "[$(date -Is)] DONE with errors" >> "$LOG"
    exit 1
fi
