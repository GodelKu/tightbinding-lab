#!/usr/bin/env bash
# Regenerate every image, JSON file and the verification table that the README refers to.
# Usage (from the repository root, after `pip install -e .`):  bash scripts/make_docs.sh
set -euo pipefail
export MPLBACKEND=Agg
OUT=docs

tblab bands chain    --out "$OUT"
tblab bands square   --path G-X-M-G --out "$OUT"
tblab bands ssh      --v 0.5 --w 1.0 --out "$OUT"
tblab bands graphene --path G-K-M-G --out "$OUT"
tblab bands haldane  --t2 0.1 --phi pi/2 --path G-K-M-Kp-G --out "$OUT"
tblab dos   square   --out "$OUT"
tblab dos   graphene --out "$OUT"
tblab ssh            --v 0.5 --w 1.0 --cells 40 --out "$OUT"
tblab berry haldane  --t2 0.1 --phi pi/2 --nk 100 --out "$OUT"
tblab phase          --t2 0.1 --n 61 --nk 24 --out "$OUT"
tblab kp             --v0 20 --well 1.0 --barrier 0.3 --out "$OUT"
tblab verify         --json "$OUT/verify.json" | tee "$OUT/verify.txt"
