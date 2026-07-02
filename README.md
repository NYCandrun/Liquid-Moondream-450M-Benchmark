# Liquid-Moondream-450M Benchmark

Benchmarking suite for the **Liquid-Moondream-450M** vision-language model.

## Overview

This repository contains scripts, datasets, and results for evaluating the
performance of the Liquid-Moondream-450M model. It is intended to track
accuracy, latency, throughput, and memory usage across tasks and hardware.

## Contents

- `benchmarks/` — benchmark scripts and task definitions
- `data/` — evaluation datasets (gitignored; see setup notes)
- `results/` — raw and summarized benchmark outputs
- `scripts/` — helper and setup utilities

## Getting Started

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Metrics Tracked

| Metric      | Description                                  |
|-------------|----------------------------------------------|
| Accuracy    | Task-level correctness on eval sets          |
| Latency     | Per-inference wall-clock time                |
| Throughput  | Samples processed per second                 |
| Memory      | Peak GPU/CPU memory during inference         |

## License

TBD.
