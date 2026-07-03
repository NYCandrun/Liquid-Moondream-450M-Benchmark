"""Enrich raw JSONL with the per-model `parsed` field (the published schema).

Reads each raw results file, applies the frozen per-model parser to
`raw_output`, and rewrites the file with `parsed` filled in. `raw_output` is
never modified, so anyone can re-derive `parsed` from the same file.
GPU-free. Usage: python3 scoring/enrich.py results/raw/bench/*.jsonl
"""

import glob
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scoring.make_dumps import parse_item  # frozen per-model parser dispatch


def enrich(path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    n_parsed = 0
    for r in rows:
        r["parsed"] = parse_item(r)
        if r["parsed"] is not None:
            n_parsed += 1
    Path(path).write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    return len(rows), n_parsed


def main():
    for pattern in sys.argv[1:]:
        for path in sorted(glob.glob(pattern)):
            n, p = enrich(path)
            print(f"{Path(path).name}: {p}/{n} parsed")


if __name__ == "__main__":
    main()
