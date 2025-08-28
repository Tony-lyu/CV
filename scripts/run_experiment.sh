#!/usr/bin/env bash
set -e

python -m src.train --config src/config/cifar10_lora.yaml
python -m src.train --config src/config/cifar10_norm.yaml
python -m src.train --config src/config/cifar10_hybrid.yaml

echo "Done. Results appended to reports/results_week1.csv"
