#!/usr/bin/env bash
# Format a review copy against the supplied original source. The clean paper
# remains authoritative; equation changes are shown without granular markup.
set -euo pipefail
original=${1:?Original manuscript .tex path required}
output=${2:?Absolute marked-output .tex path required}
cd "$(dirname "$0")"
latexdiff --graphics-markup=none --math-markup=0 \
  --disable-citation-markup --type=CFONT --exclude-safecmd=label \
  "$original" paper.tex |
  perl -0pe 's/Representation\/model/Representation\/\\allowbreak model/g;
    s/\\DIFdelbegin \\bibitem\{graboxyContinuous\}/\\DIFdelbegin \\bibitem{graboxyContinuousOriginal}/g;
    s/acien2020BeCAPTCHA/acien2022BeCAPTCHA/g;
    s/\\maketitle/\\maketitle\n\\noindent\\textbf{Revision markup.} Blue text indicates additions; small red text indicates deletions relative to the original supplied manuscript. Displayed mathematics and the new comparison table are shown as whole blocks, without granular internal markup.\\par\\medskip/' > "$output"
# The positioning table is entirely new. Treat it as one added block rather
# than permitting latexdiff to align unrelated old prose with its cells.
node - "$output" <<'JS'
const fs=require('fs');
const output=process.argv[2];
const pattern=/\\begin\{table\*\}[\s\S]*?\\end\{table\*\}/g;
const key='\\label{tab:related_work_positioning}';
const clean=fs.readFileSync('paper.tex','utf8');
const added=[...clean.matchAll(pattern)].find(m=>m[0].includes(key))[0];
const marked=fs.readFileSync(output,'utf8');
const old=[...marked.matchAll(pattern)].find(m=>m[0].includes(key))[0];
fs.writeFileSync(output,marked.replace(old,
  added.replace('\\centering','\\centering\n\\color{blue}')));
JS
