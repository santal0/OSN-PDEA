#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
src="${1:-$script_dir/main.tex}"
outdir="${2:-$script_dir/build}"

if [[ "${1:-}" != "" && "$src" != /* ]]; then
  src="$PWD/$src"
fi

if [[ "${2:-}" != "" && "$outdir" != /* ]]; then
  outdir="$PWD/$outdir"
fi

if [[ ! -f "$src" ]]; then
  echo "error: source file not found: $src" >&2
  exit 1
fi

mkdir -p "$outdir"

latexmk \
  -xelatex \
  -bibtex \
  -interaction=nonstopmode \
  -halt-on-error \
  -file-line-error \
  -synctex=1 \
  -outdir="$outdir" \
  -recorder \
  "$src"

pdf_name="$(basename "${src%.tex}").pdf"
echo "built: $outdir/$pdf_name"
