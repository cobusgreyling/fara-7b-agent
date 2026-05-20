#!/usr/bin/env bash
#
# Download Fara-7B Q4_K_M quantisation and vision projector from
# Hugging Face. Skips files that already exist.
#
set -euo pipefail

REPO="bartowski/microsoft_Fara-7B-GGUF"
MODEL_FILE="microsoft_Fara-7B-Q4_K_M.gguf"
MMPROJ_FILE="mmproj-microsoft_Fara-7B-f16.gguf"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${REPO_ROOT}/models"
mkdir -p "${DEST}"

base="https://huggingface.co/${REPO}/resolve/main"

download() {
    local remote="$1"
    local local_name="$2"
    local target="${DEST}/${local_name}"
    if [[ -f "${target}" ]]; then
        echo "[skip] ${local_name} already exists"
        return
    fi
    echo "[get ] ${remote} -> ${target}"
    curl -L --fail --progress-bar -o "${target}" "${base}/${remote}?download=true"
}

download "${MODEL_FILE}" "model.gguf"
download "${MMPROJ_FILE}" "mmproj.gguf"

echo
echo "Done. Model files in ${DEST}:"
ls -lh "${DEST}"
