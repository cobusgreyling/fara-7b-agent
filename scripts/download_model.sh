#!/usr/bin/env bash
#
# Download a Fara-7B GGUF quantisation and the vision projector from
# Hugging Face. Skips files that already exist.
#
# Usage:
#   scripts/download_model.sh                  # Q4_K_M (default, ~4.7 GB)
#   scripts/download_model.sh --quant Q5_K_M   # ~5.4 GB, higher fidelity
#   scripts/download_model.sh --quant Q8_0     # ~8.1 GB, full-precision-ish;
#                                              # eliminates wrapper-name drift
#                                              # in the parser.
#
# Quantisations available at bartowski/microsoft_Fara-7B-GGUF as of
# November 2025: Q3_K_S, Q3_K_M, Q3_K_L, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M,
# Q6_K, Q8_0, f16. This script accepts any of those tags.
#
set -euo pipefail

QUANT="Q4_K_M"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quant)
            QUANT="$2"
            shift 2
            ;;
        --quant=*)
            QUANT="${1#--quant=}"
            shift
            ;;
        -h|--help)
            sed -n '1,15p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

REPO="bartowski/microsoft_Fara-7B-GGUF"
MODEL_FILE="microsoft_Fara-7B-${QUANT}.gguf"
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

echo "Quantisation: ${QUANT}"
download "${MODEL_FILE}" "model.gguf"
download "${MMPROJ_FILE}" "mmproj.gguf"

echo
echo "Done. Model files in ${DEST}:"
ls -lh "${DEST}"
