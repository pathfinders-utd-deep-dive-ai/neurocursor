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
    s/\\maketitle/\\maketitle\n\\noindent\\textbf{Revision markup.} Blue text indicates additions; small red text indicates deletions relative to the original supplied manuscript. Displayed mathematics and selected revised tables are shown as whole blocks, without granular internal markup. The clean manuscript is authoritative.\\par\\medskip/' > "$output"
# The positioning table is entirely new. Treat it as one added block rather
# than permitting latexdiff to align unrelated old prose with its cells.
node - "$output" <<'JS'
const fs=require('fs');
const output=process.argv[2];
const pattern=/\\begin\{(table\*?)\}[\s\S]*?\\end\{\1\}/g;
const clean=fs.readFileSync('paper.tex','utf8');
let marked=fs.readFileSync(output,'utf8');
for(const label of ['tab:related_work_positioning','tab:rq_outcomes','tab:operating_points','tab:split_comparison']) {
  const key='\\label{'+label+'}';
  const added=[...clean.matchAll(pattern)].find(m=>m[0].includes(key))[0];
  const old=[...marked.matchAll(pattern)].find(m=>m[0].includes(key))[0];
  marked=marked.replace(old,added.replace('\\centering','\\centering\n\\color{blue}'));
}
fs.writeFileSync(output,marked);
JS
