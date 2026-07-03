"""Scoring package for the edge-VLM benchmark (shared by bench and sweep parts).

Pure Python standard library only -- no numpy/torch/PIL -- so scoring runs
anywhere, including machines without a GPU or any ML stack installed.

Modules
-------
metrics : frozen, pure metric functions (IoU, grounding/POPE/count scoring).
score   : CLI that scores raw JSONL run files into ``.scored.json`` reports.
"""
