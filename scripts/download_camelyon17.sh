#!/usr/bin/env bash
# Downloads Camelyon17-WILDS v1.0 (~10.7 GB compressed) into data/camelyon17_v1.0.
# Uses curl rather than the WILDS downloader, which fails on macOS python.org builds
# without installed SSL certificates. Re-running resumes a partial download.
set -euo pipefail

URL="https://huggingface.co/datasets/pangwei/wilds-camelyon17/resolve/2ed40cf44d4e63de0e89f756e5ef95108127c1df/camelyon17_v1.0.tar.gz"
DEST="${1:-data}/camelyon17_v1.0"
ARCHIVE="$DEST/archive.tar.gz"

mkdir -p "$DEST"
if [ -f "$DEST/RELEASE_v1.0.txt" ]; then
  echo "Already present: $DEST"
  exit 0
fi

CURL=curl
[ -x /usr/bin/curl ] && CURL=/usr/bin/curl  # conda's curl on CARC can lack HTTPS support
$CURL -L --fail --retry 5 -C - -o "$ARCHIVE" "$URL"
echo "Extracting..."
tar -xzf "$ARCHIVE" -C "$DEST"
rm "$ARCHIVE"
echo "Done: $DEST"
