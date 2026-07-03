"""Subprocess entrypoint for one sweep cell inside the SWEEPBOX container.

Runs in a plain pinned venv (/opt/tf5 or /opt/tf457) — no modal import.
Reads {"config", "prompt", "results_base"} as JSON on stdin; runs the shared
cell core; prints a one-line summary marker (without the bulky rows) to
stdout. The driver (run_sweep_serial.py) reads rows from the volume file and
performs volume commits.
"""

import json
import sys

sys.path.insert(0, "/root")  # modal mounts local python source at /root

from harness.cell_core import run_cell_core  # noqa: E402


def main():
    payload = json.load(sys.stdin)
    summary = run_cell_core(payload["config"], payload["prompt"], payload["results_base"])
    summary.pop("rows_jsonl", None)
    print("SUMMARY_JSON:" + json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
