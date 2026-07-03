"""Export a full `pip freeze` lockfile from inside every Modal image.

Committed to envs/locks/<env>.lock.txt — the environment pin the brief requires.
Re-run whenever an image definition changes (and re-pin envs/images.py from these
locks before the Phase 3 freeze).

Usage (from repo root):
  .venv/bin/modal run harness/export_locks.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import modal

from envs.images import DATASETS_IMAGE, MOONDREAM2_IMAGE, TF457_IMAGE, TF5_IMAGE

app = modal.App("edge-vlm-phase0-export-locks")


def _freeze():
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"], capture_output=True, text=True, check=True
    )
    return proc.stdout


@app.function(image=TF5_IMAGE, timeout=600)
def lock_tf5():
    return _freeze()


@app.function(image=TF457_IMAGE, timeout=600)
def lock_tf457():
    return _freeze()


@app.function(image=MOONDREAM2_IMAGE, timeout=600)
def lock_moondream2():
    return _freeze()


@app.function(image=DATASETS_IMAGE, timeout=600)
def lock_datasets():
    return _freeze()


@app.local_entrypoint()
def main():
    calls = {
        "tf5": lock_tf5.spawn(),
        "tf457": lock_tf457.spawn(),
        "moondream2": lock_moondream2.spawn(),
        "datasets": lock_datasets.spawn(),
    }
    lock_dir = REPO_ROOT / "envs" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    for name, call in calls.items():
        text = call.get()
        (lock_dir / f"{name}.lock.txt").write_text(text)
        print(f"wrote envs/locks/{name}.lock.txt ({len(text.splitlines())} packages)")
