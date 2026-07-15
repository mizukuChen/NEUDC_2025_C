#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT_DIR}/.venv/bin/python"
LOG_DIR="${ROOT_DIR}/runtime/logs"
LOG_FILE="${LOG_DIR}/camera_debug.log"
LOCK_FILE="${LOG_DIR}/camera_debug_gui.lock"

mkdir -p "${LOG_DIR}"
if [[ ! -x "${PYTHON}" ]]; then
    printf 'ERROR: virtual environment Python not found: %s\n' "${PYTHON}" >>"${LOG_FILE}"
    exit 1
fi

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    printf '[%s] camera debug GUI is already running; ignoring duplicate start\n' \
        "$(date '+%Y-%m-%d %H:%M:%S')" >>"${LOG_FILE}"
    exit 0
fi

cd "${ROOT_DIR}"
printf '\n[%s] starting camera debug GUI\n' "$(date '+%Y-%m-%d %H:%M:%S')" >>"${LOG_FILE}"
exec "${PYTHON}" -m src.camera_debug \
    --camera-index 0 \
    --width 1920 \
    --height 1080 \
    "$@" >>"${LOG_FILE}" 2>&1
