#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin?download=true"
OUT="${1:-ggml-large-v3-turbo.bin}"
TMP="${OUT}.part"

install_hub_cache() {
    local src="$1"
    local cache="${HOME}/.makepad/weights/stt"
    local dest="${cache}/ggml-large-v3-turbo.bin"
    mkdir -p "${cache}"
    if [[ -e "${dest}" ]]; then
        return 0
    fi
    ln "${src}" "${dest}" 2>/dev/null \
        || ln -s "$(cd "$(dirname "${src}")" && pwd)/$(basename "${src}")" "${dest}"
    echo "linked ${dest}"
}

if [[ -f "${OUT}" ]]; then
    echo "Model already exists: ${OUT}"
    install_hub_cache "${OUT}"
    exit 0
fi

echo "Downloading ${OUT}..."
curl -L --fail --retry 3 --retry-delay 2 -o "${TMP}" "${URL}"
mv "${TMP}" "${OUT}"
echo "Done: ${OUT}"
install_hub_cache "${OUT}"
