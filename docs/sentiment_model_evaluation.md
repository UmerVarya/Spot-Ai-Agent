# Sentiment Model Evaluation Playbook

The fused sentiment stack now includes FinLlama and FinGPT with weights that
default to 0.50 / 0.50.  To keep the fusion aligned with the best-performing
language models release-over-release, run the following evaluation workflow
whenever a new financial LLM becomes available (e.g. Llama 3-70B finetunes or
Mistral-FinRL updates).

## 1. Build a validation set

* Collect at least 200 dated macro and micro news snippets with ground truth
  sentiment scores in ``[-1, 1]``.  Regulatory filings, FOMC statements and
  CPI prints are particularly valuable because they stress long-form numeric
  reasoning – an area where FinGPT excels compared with earlier classifiers.
* Store the dataset as JSON lines with the fields ``{"headlines": [...], "label": float}``.

## 2. Run the evaluation helper

```python
from pathlib import Path
import json

from fused_sentiment import calibrate_fusion_weights, evaluate_models

path = Path("validation_macro_news.jsonl")
validation = [
    (item["headlines"], float(item["label"]))
    for item in (json.loads(line) for line in path.read_text().splitlines())
]

metrics = evaluate_models(validation)
print("Per-model MAE:", {k: round(v["mae"], 3) for k, v in metrics.items()})

new_weights = calibrate_fusion_weights(validation, step=0.05)
print("Suggested fusion weights:", new_weights)
```

The coarse simplex search defaults to 0.05 increments.  Reduce the step to
0.02 once enough data is available to justify finer calibration.

## 3. Update the production weights

If the suggested weights materially differ from the defaults, persist them in
configuration (for example via an environment variable or JSON file) and pass
``fusion_weights=...`` into ``analyze_headlines``.  This keeps FinGPT or future
models (FinGPT 2025, Llama 3-70B finetuned, Mistral-FinRL, etc.) in lockstep
with real-world validation performance.

## 4. Archive results

Log the evaluation metrics, chosen weights and dataset revision inside
``docs/evaluations/YYYY-MM-DD-sentiment.json`` so future calibrations can track
progression across model upgrades.
