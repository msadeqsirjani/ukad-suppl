#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/plot_efficiency.py
python scripts/plot_pareto.py
python scripts/plot_activations.py
python scripts/plot_activations.py --main --out docs/paper/iclr2027-9page/figures/activations_main.png
python scripts/plot_qualitative.py --main
python scripts/plot_displacement.py
python scripts/plot_skip_gates.py
