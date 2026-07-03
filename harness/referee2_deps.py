"""Discover the FULL set of undeclared eager-import deps the newer VLMEvalKit
needs to `import vlmeval` (its dataset/__init__ imports ~60 submodules, several
of which import packages not in requirements.txt — e.g. sklearn). One container
run finds them all so we bake the exact list into the images, instead of many
rebuild rounds.

Usage: .venv/bin/modal run harness/referee2_deps.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import modal

from envs.images import HF_CACHE_PATH, VLMEVAL2_TF457_IMAGE, hf_cache_volume

app = modal.App("edge-vlm-referee2-deps")

# module import name -> pip package name (only where they differ)
NAME_MAP = {"sklearn": "scikit-learn", "cv2": "opencv-python", "PIL": "pillow",
            "yaml": "pyyaml", "skimage": "scikit-image"}


@app.function(image=VLMEVAL2_TF457_IMAGE, timeout=2400,
              volumes={HF_CACHE_PATH: hf_cache_volume})
def discover():
    import importlib
    import subprocess

    installed = []
    for _ in range(40):
        for k in list(sys.modules):
            if k.startswith("vlmeval"):
                del sys.modules[k]
        importlib.invalidate_caches()
        try:
            from vlmeval.config import supported_VLM  # noqa: F401
            return {"ok": True, "installed": installed}
        except ModuleNotFoundError as exc:
            mod = (exc.name or "").split(".")[0]
            if not mod:
                return {"ok": False, "installed": installed, "err": repr(exc)}
            pkg = NAME_MAP.get(mod, mod)
            r = subprocess.run(["pip", "install", "--no-cache-dir", pkg],
                               capture_output=True, text=True)
            installed.append({"module": mod, "pkg": pkg, "rc": r.returncode})
            if r.returncode != 0:
                return {"ok": False, "installed": installed,
                        "fail_pkg": pkg, "err": r.stderr[-1200:]}
        except Exception:
            import traceback
            return {"ok": False, "installed": installed,
                    "other": traceback.format_exc()[-1500:]}
    return {"ok": False, "installed": installed, "note": "iteration cap hit"}


@app.local_entrypoint()
def main():
    print(json.dumps(discover.remote(), indent=1))
