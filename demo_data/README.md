# Image-detail lookup table

`image_token_lookup.csv` and `image_token_lookup.json` hold the same 17 rows. Each row gives the accuracy, speed and (where measured) energy of one model at one image-detail setting. Regenerate them with `python3 -m scoring.demo_table`. Every number is computed from `results/raw/` and `results/probe_energy.json`.

| Column | Meaning |
|---|---|
| `source` | `sweep` is the full test set. `probe` is the 100-question counting subset with measured energy. `shipped` is Moondream 0.5B in its own runtime. |
| `model`, `task` | The model and the task it was run on. LFM is measured on find-object. SmolVLM2 is measured on counting because it cannot produce boxes. |
| `setting_kind`, `setting` | The dial that was changed: image tokens for LFM (minimum and maximum set equal), longest image edge in pixels for SmolVLM2. Moondream 0.5B has no dial and appears at its fixed 729 tokens. |
| `hardware` | The GPU or CPU the row was measured on. Speed is only comparable between rows with the same hardware. |
| `n_items` | How many questions the row covers. |
| `accuracy`, `accuracy_metric` | Fraction correct. `Acc@0.5` means the predicted box overlaps the true box by at least half. `exact match` means the number is exactly right. |
| `median_ms_per_item` | Median time to answer one picture, in milliseconds, with the model already loaded. |
| `mean_input_tokens` | Average number of tokens fed to the model, image and question text together. |
| `items_upscaled_by_fallback` | LFM only. Pictures that were enlarged locally so the model's resizing code did not crash. They are scored normally. |
| `j_per_answer`, `j_per_answer_net_of_idle` | Energy per answer from the GPU's own counter. The second column subtracts the GPU's idle power. Filled for `probe` rows only. |
| `note` | Caveats for the row. |

The LFM and SmolVLM2 accuracy columns measure different tasks and must not be plotted against each other.
