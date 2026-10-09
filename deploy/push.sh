#!/usr/bin/env bash
# Run on your machine. Copies the app code to the server. Data, api_keys.txt,
# .env and the sessions DB are never touched; the previous code is archived
# to /root/pubmed_search_backups/ first, so it can be restored.
#
#   bash deploy/push.sh            then on the server:   bash deploy/install.sh
set -euo pipefail
HOST="${MSS_HOST:-root@152.53.80.217}"
DEST=/root/pubmed_search
cd "$(dirname "$0")/.."

stamp=$(date +%Y%m%d-%H%M%S)
ssh "$HOST" "mkdir -p /root/pubmed_search_backups && \
  tar czf /root/pubmed_search_backups/code-$stamp.tgz -C $DEST --exclude='*.db' --exclude=.git . && \
  echo 'Previous code archived to /root/pubmed_search_backups/code-$stamp.tgz'"
rsync -av --exclude-from=deploy/rsync-exclude.txt ./ "$HOST:$DEST/"
echo
echo "Code copied. To switch the services, run on the server:"
echo "  cd $DEST && bash deploy/install.sh"
