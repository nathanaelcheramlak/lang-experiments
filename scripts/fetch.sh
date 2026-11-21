#!/usr/bin/env bash
set -euo pipefail

root=$(dirname "$(dirname "$(realpath "$0")")")

download() {
  local url=$1 dest=$2
  if [[ ! -s "$dest" ]]; then
    curl -fL --retry 3 -C - "$url" -o "$dest.part"
    mv "$dest.part" "$dest"
  fi
}

case ${1:-} in
  sagalee)
    raw="$root/data/raw/sagalee/openslr-157"
    mkdir -p "$raw"
    cd "$raw"
    download "${SAGALEE_URL:-https://openslr.trmal.net/resources/157/sagalee.7z}" sagalee.7z
    if [[ -f archive.sha256 ]]; then
      sha256sum --check archive.sha256
    else
      sha256sum sagalee.7z > archive.sha256
    fi
    if [[ ! -d extracted ]]; then
      staging=$(mktemp -d .extract.XXXXXX)
      trap 'rm -rf -- "$staging"' EXIT
      python -m py7zr x sagalee.7z "$staging"
      mv -T "$staging" extracted
    fi
    printf 'Sagalee extracted under %s/extracted\n' "$raw"
    ;;
  large-v3)
    mkdir -p "$root/models/openai-whisper"
    cd "$root/models/openai-whisper"
    sha=e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb
    download "https://openaipublic.azureedge.net/main/whisper/models/$sha/large-v3.pt" large-v3.pt
    printf '%s  large-v3.pt\n' "$sha" | sha256sum --check
    ;;
  *) printf 'Usage: bash scripts/fetch.sh {sagalee|large-v3}\n' >&2; exit 2 ;;
esac
