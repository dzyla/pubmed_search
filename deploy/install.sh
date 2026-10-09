#!/usr/bin/env bash
# Run ON THE SERVER as root, from /root/pubmed_search, after deploy/push.sh.
# Switches from the old three services (model_api, search_api, streamlit_app)
# to two: mss-backend (model + index + REST + MCP) and mss-ui (Streamlit).
#
#   bash deploy/install.sh              install and switch
#   bash deploy/install.sh --rollback   go back to the old services
#
# Idempotent: safe to run again after a code update (it restarts both services).
set -euo pipefail
APP=/root/pubmed_search
PY=/root/space_env/bin
cd "$APP"

if [[ "${1:-}" == "--rollback" ]]; then
  systemctl disable --now mss-ui mss-backend 2>/dev/null || true
  for f in /etc/nginx/sites-available/*.pre-mss; do
    [[ -e "$f" ]] && cp "$f" "${f%.pre-mss}"
  done
  nginx -t && systemctl reload nginx
  systemctl enable --now model_api search_api streamlit_app
  echo "Old services restored. To restore the old code as well:"
  echo "  ls /root/pubmed_search_backups/   then   tar xzf <archive> -C $APP"
  exit 0
fi

echo "== 1/5 Python packages (adds only what is missing; torch is left alone)"
"$PY/pip" install --quiet "mcp==1.26.0" "httpx==0.28.1"
"$PY/python" -c "import mcp, httpx, faiss, sentence_transformers, streamlit, google.genai" \
  || { echo "A required package is missing in $PY"; exit 1; }

echo "== 2/5 Environment file"
if [[ ! -f .env ]]; then
  (umask 077; sed "s/^MSS_INTERNAL_KEY=.*/MSS_INTERNAL_KEY=$(openssl rand -hex 32)/" deploy/env.example > .env)
  echo "Created .env with a new internal key"
fi

echo "== 3/5 Services"
install -m 644 deploy/systemd/mss-backend.service deploy/systemd/mss-ui.service /etc/systemd/system/
systemctl daemon-reload
# The old services hold ports 8000/8080/8501 and a second copy of the index.
systemctl disable --now streamlit_app search_api model_api 2>/dev/null || true
systemctl enable mss-backend mss-ui
systemctl restart mss-backend
echo -n "Waiting for the backend (model + index)"
for _ in $(seq 1 120); do
  curl -sf http://127.0.0.1:8080/health >/dev/null && break
  echo -n "."; sleep 5
done
echo
if ! curl -sf http://127.0.0.1:8080/health; then
  echo "Backend is not healthy. Check: journalctl -u mss-backend -n 100"
  echo "To go back: bash deploy/install.sh --rollback"
  exit 1
fi
echo
systemctl restart mss-ui

echo "== 4/5 nginx (rate limits, API/MCP routes, maintenance page)"
install -d /var/www/mss
install -m 644 deploy/maintenance.html /var/www/mss/
install -m 644 deploy/nginx/mss-ratelimit.conf /etc/nginx/conf.d/
install -m 644 deploy/nginx/mss-locations.conf /etc/nginx/snippets/
for pair in "manuscript-search.org:site-manuscript-search.org.conf" "mssearch.org:site-msssearch.org.conf"; do
  site="/etc/nginx/sites-available/${pair%%:*}"
  [[ -e "$site.pre-mss" ]] || cp "$site" "$site.pre-mss"
  install -m 644 "deploy/nginx/${pair##*:}" "$site"
done
if nginx -t; then
  systemctl reload nginx
else
  echo "nginx config test failed — restoring previous site files"
  for f in /etc/nginx/sites-available/*.pre-mss; do cp "$f" "${f%.pre-mss}"; done
  rm -f /etc/nginx/conf.d/mss-ratelimit.conf
  nginx -t && systemctl reload nginx
  exit 1
fi

echo "== 5/5 Smoke test"
curl -sf http://127.0.0.1:8080/v1/stats | head -c 300; echo
curl -sf -o /dev/null -w "UI: HTTP %{http_code}\n" http://127.0.0.1:8501/
echo "Done. Logs: journalctl -u mss-backend -f   |   journalctl -u mss-ui -f"
