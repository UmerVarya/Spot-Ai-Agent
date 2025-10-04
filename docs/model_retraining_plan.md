# Microstructure Feature Retraining Plan

The probability model now ingests an expanded set of microstructure fields
(order imbalance, cumulative delta, taker participation, spoofing signals,
etc.).  To ensure the classifier learns meaningful coefficients for the new
inputs, follow the retraining workflow below once sufficient data has been
captured.

## 1. Start Collecting Enriched Trade Logs

* Ensure the live or paper trading agent is running with the latest code so
  every completed trade writes the new microstructure metrics to
  `trade_history.csv` via `trade_storage.log_trade_result`.
* Monitor the log warnings emitted by `ml_model.train_model`.  The training
  routine reports a low coverage warning when fewer than 10 % of samples have a
  non-zero value for a given microstructure signal.  Continue operating until
  these warnings disappear for the fields you intend to include.

## 2. Build a Clean Training Dataset

* Use `scripts/prepare_dataset.py` to normalise column names, coerce types and
  generate train/validation/test splits with the enriched feature set.
* Inspect the resulting dataset to verify that the microstructure columns are
  populated and have sensible ranges (e.g. `order_flow_score` near ±1,
  `spread_bps` in tens of basis points).

## 3. Retrain the Classifier

* Invoke `ml_model.train_model()` (or run the corresponding CLI wrapper if you
  have one) to retrain using the new dataset.
* The training summary now records `micro_feature_stats` which quantify the
  non-zero coverage and standard deviation of each microstructure signal.  Keep
  these numbers for future reference.
* Evaluate the reported validation metrics (accuracy, ROC-AUC, Brier score).
  Compare them against the previous production run to confirm the enriched
  features are beneficial.

## 4. Validate Before Deployment

* Perform out-of-sample checks: replay a recent time window (forward testing) or
  run a short paper-trading session to confirm the probability outputs behave as
  expected.
* Review feature importances in `ml_model.json` to understand which
  microstructure signals drive decisions.  If certain features remain unused
  (very low importance and coverage), consider pruning them or improving their
  data quality before the next training cycle.

## 5. Schedule Ongoing Refreshes

* Plan for periodic retraining (e.g. monthly or after every N new trades) so
  the classifier adapts to regime changes while continuing to leverage the
  microstructure telemetry.
* Archive artefacts (`ml_model.pkl`, `ml_model.json`, dataset snapshots) from
  each training run to support regression analysis and rollbacks.

Following this cadence ensures the live agent benefits from the additional
order-flow context without regressing on model stability or calibration.
