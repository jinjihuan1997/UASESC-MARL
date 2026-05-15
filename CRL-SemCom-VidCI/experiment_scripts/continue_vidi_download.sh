#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_ROOT="${VIDI_RAW_ROOT:-$REPO_ROOT/data/VIDI/raw}"
CSV_PATH="${VIDI_CSV_PATH:-$REPO_ROOT/data/VIDI/metadata/vidi_all.csv}"
TMP_DIR="${VIDI_TMP_DIR:-/tmp/vidi_tmp}"
OFFICIAL_SCRIPT="${VIDI_OFFICIAL_SCRIPT:-/tmp/VIDI_official/download.py}"
YTDLP_WRAPPER_DIR="${VIDI_YTDLP_WRAPPER_DIR:-/tmp/vidi_bin}"
JOBS="${VIDI_JOBS:-8}"
SLEEP_SECONDS="${VIDI_RETRY_SLEEP_SECONDS:-15}"
LOG_PATH="${VIDI_LOG_PATH:-$REPO_ROOT/data/VIDI/download_loop.log}"

mkdir -p "$RAW_ROOT" "$TMP_DIR"

count_mp4() {
  find "$RAW_ROOT" -type f -name '*.mp4' | wc -l
}

echo "[$(date '+%F %T')] Starting VIDI download loop" | tee -a "$LOG_PATH"
echo "[$(date '+%F %T')] raw_root=$RAW_ROOT csv=$CSV_PATH jobs=$JOBS tmp_dir=$TMP_DIR" | tee -a "$LOG_PATH"

while true; do
  before_count="$(count_mp4)"
  echo "[$(date '+%F %T')] pass_start current_mp4=$before_count" | tee -a "$LOG_PATH"

  set +e
  PATH="$YTDLP_WRAPPER_DIR:$PATH" python "$OFFICIAL_SCRIPT" "$CSV_PATH" "$RAW_ROOT" -n "$JOBS" -t "$TMP_DIR" >> "$LOG_PATH" 2>&1
  exit_code=$?
  set -e

  after_count="$(count_mp4)"
  delta=$((after_count - before_count))
  echo "[$(date '+%F %T')] pass_end exit_code=$exit_code current_mp4=$after_count delta=$delta" | tee -a "$LOG_PATH"

  sleep "$SLEEP_SECONDS"
done
