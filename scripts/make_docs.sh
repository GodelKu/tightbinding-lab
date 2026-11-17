#!/usr/bin/env bash
# Regenerate every image and JSON file in docs/ and the generated blocks of README.md.
# Usage (from the repository root, after `pip install -e ".[dev]"`):  bash scripts/make_docs.sh
# The mutation check (tools/mutation_check.py) is slower and is run separately.
set -euo pipefail
export MPLBACKEND=Agg
PYTHON="${PYTHON:-python}"
OUT=docs

"$PYTHON" -m tblab bands chain    --out "$OUT"
"$PYTHON" -m tblab bands square   --path G-X-M-G --out "$OUT"
"$PYTHON" -m tblab bands ssh      --v 0.5 --w 1.0 --out "$OUT"
"$PYTHON" -m tblab bands graphene --path G-K-M-G --out "$OUT"
"$PYTHON" -m tblab bands haldane  --t2 0.1 --phi pi/2 --path G-K-M-Kp-G --out "$OUT"
"$PYTHON" -m tblab dos   square   --out "$OUT"
"$PYTHON" -m tblab dos   graphene --out "$OUT"
"$PYTHON" -m tblab ssh            --v 0.5 --w 1.0 --cells 40 --out "$OUT"
"$PYTHON" -m tblab berry haldane  --t2 0.1 --phi pi/2 --nk 100 --out "$OUT"
"$PYTHON" -m tblab phase          --t2 0.1 --n 61 --nk 24 --out "$OUT"
"$PYTHON" -m tblab kp             --v0 20 --well 1.0 --barrier 0.3 --out "$OUT"
"$PYTHON" -m tblab verify         --json "$OUT/verify.json" | tee "$OUT/verify.txt"
"$PYTHON" tools/measure.py
"$PYTHON" tools/fill_readme.py
