"""Probe preflight (change #12): which pool items COMPLETE at every dial setting?

Runs inside the probe container, in one pinned venv, on the probe's own GPU
die. For one model it attempts a real 1-token generate (the standard adapter
path) for every pool item at every budget, and reports pass/fail per
(item, budget). The driver intersects both models' passes and fixes the probe
question set to items every cell can complete — the efficiency curve then
compares identical work across all 8 cells. Crash-proneness of forced budgets
is a separate, disclosed finding (measured at full scale by the Part 2 sweep).

Reads JSON on stdin: {"model", "budgets", "pool_n", "seed"}
Prints one line:    PREFLIGHT_JSON:{"model", "pool" (ordered ids),
                    "pass" {item_id: [ok-per-budget]}, "fail_examples"}
"""

import json
import sys

sys.path.insert(0, "/root")

from harness import data as hdata  # noqa: E402
from harness.adapters import BUDGET_WIRING, MODEL_PINS, ITTAdapter  # noqa: E402


def main():
    spec = json.load(sys.stdin)
    model_key, budgets = spec["model"], spec["budgets"]
    pin = MODEL_PINS[model_key]

    items = hdata.load_items("count")
    pool, _ = hdata.fixed_sample(items, int(spec["pool_n"]), int(spec["seed"]), "count")

    # One model load; budget applied per round the same way the adapter does.
    adapter = ITTAdapter(pin["model_id"], pin["revision"],
                         trust_remote_code=pin.get("trust_remote_code", False))
    wiring = BUDGET_WIRING[model_key]

    passes = {it["item_id"]: [] for it in pool}
    fail_examples = []
    for budget in budgets:
        wired = wiring(budget)
        adapter.template_kwargs = wired.get("template", {})
        if wired.get("pretrained"):
            from transformers import AutoProcessor
            adapter.processor = AutoProcessor.from_pretrained(
                pin["model_id"], revision=pin["revision"],
                trust_remote_code=pin.get("trust_remote_code", False),
                **wired["pretrained"],
            )
        for it in pool:
            try:
                adapter.generate(it["image"], "preflight", 1, task="count")
                passes[it["item_id"]].append(True)
            except Exception as exc:
                passes[it["item_id"]].append(False)
                if len(fail_examples) < 10:
                    fail_examples.append({"item_id": it["item_id"], "budget": budget,
                                          "error": repr(exc)[:200]})

    print("PREFLIGHT_JSON:" + json.dumps({
        "model": model_key,
        "budgets": budgets,
        "pool": [it["item_id"] for it in pool],
        "pass": passes,
        "fail_examples": fail_examples,
    }))


if __name__ == "__main__":
    main()
