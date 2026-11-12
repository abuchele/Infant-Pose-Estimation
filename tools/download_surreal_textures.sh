#!/usr/bin/env bash
# Download only the SURREAL clothing textures (textures.tar.gz; same 935 jpgs as listed in
# surreal/download/files/files_smpl_data.txt), not smpl_data.npz (2.5 GB, not needed here).
#
# Needs SURREAL_USER and SURREAL_PASSWORD in .env (credentials e-mailed by SURREAL after
# accepting the licence at https://www.di.ens.fr/willow/research/surreal/data/).
# The password is passed to curl on stdin (--config -), never on the command line or in output.
#
#   bash tools/download_surreal_textures.sh            # -> syn_generation/render/textures/surreal/
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=syn_generation/render/textures/surreal
URL=https://lsh.paris.inria.fr/SURREAL/smpl_data/textures.tar.gz

user=$(grep -E '^SURREAL_USER=' .env | head -1 | cut -d= -f2- || true)
pass=$(grep -E '^SURREAL_PASSWORD=' .env | head -1 | cut -d= -f2- || true)
if [ -z "$user" ] || [ -z "$pass" ]; then
  echo "SURREAL_USER / SURREAL_PASSWORD not set in .env" >&2; exit 1
fi
if [ -e "$OUT" ]; then echo "$OUT exists; not overwriting" >&2; exit 1; fi

DL=$(mktemp -d "${TMPDIR:-/tmp}/surreal_tex.XXXXXX")
trap 'rm -rf "$DL"' EXIT
printf 'user = "%s:%s"\n' "$user" "$pass" | \
  curl --config - --fail --location --show-error --progress-bar -o "$DL/textures.tar.gz" "$URL"
unset pass

# list first, refuse absolute paths or '..' before extracting (untrusted archive)
if tar -tzf "$DL/textures.tar.gz" | grep -qE '(^/|(^|/)\.\.(/|$))'; then
  echo "archive contains unsafe paths; not extracting" >&2; exit 1
fi
mkdir -p "$DL/x"
tar -xzf "$DL/textures.tar.gz" -C "$DL/x"
mkdir -p "$(dirname "$OUT")"
src=$(find "$DL/x" -maxdepth 2 -type d -name textures | head -1)
mv "${src:-$DL/x}" "$OUT"
md5 -q "$DL/textures.tar.gz" 2>/dev/null | sed 's/^/textures.tar.gz md5 /' || md5sum "$DL/textures.tar.gz"
echo "images: $(find "$OUT" -type f \( -iname '*.jpg' -o -iname '*.png' \) | wc -l) in $OUT"
