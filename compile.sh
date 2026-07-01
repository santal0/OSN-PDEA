#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
src="${script_dir}/main.tex"  # 固定源文件路径
outdir="${script_dir}/build"   # 固定输出目录

if [[ ! -f "$src" ]]; then
  echo "error: source file not found: $src" >&2
  exit 1
fi

mkdir -p "$outdir"

# 使用 "$@" 传递所有额外的命令行参数给 latexmk
latexmk \
  -xelatex \
  -bibtex \
  -interaction=nonstopmode \
  -halt-on-error \
  -file-line-error \
  -synctex=1 \
  -outdir="$outdir" \
  -recorder \
  "$@" \
  "$src"

pdf_name="$(basename "${src%.tex}").pdf"
out_pdf="$outdir/$pdf_name"
root_pdf="$script_dir/$pdf_name"

cp "$out_pdf" "$root_pdf"

echo "built: $out_pdf"
echo "synced: $root_pdf"
