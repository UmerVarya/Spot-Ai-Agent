# Reinforcement Learning from Human Feedback for Trade Alignment

This document outlines an exploratory roadmap for adapting reinforcement learning from human feedback (RLHF) to improve the Spot AI Agent's trade selection heuristics.

## Motivation

The current agent relies on rule-based veto layers and deterministic scoring models. While these components ensure basic risk discipline, they do not learn directly from the realised profit and loss (P&L) of previous trades. RLHF offers a path to align the model with the objective of maximizing long-horizon trading performance instead of merely generating plausible rationales.

## High-level approach

1. **Scenario generation** – Assemble a corpus of historical trade opportunities. Each scenario should include the market snapshot, relevant news snippets, technical indicators, and the ultimate realised P&L if a trade had been executed.
2. **Policy rollouts** – Simulate the existing agent (or heuristic baselines) on each scenario to obtain proposed actions (approve/reject, sizing, stop-loss, etc.).
3. **Reward modeling** – Train a reward model that scores a dialogue or decision trace based on the eventual trade outcome. Positive reward corresponds to trades that would have improved the equity curve; negative reward penalises avoidable losses.
4. **Preference dataset** – Convert the reward signals into pairwise or scalar preferences. For example, dialogue ``A`` that leads to a +3% return should outrank dialogue ``B`` that draws down -2% on the same scenario.
5. **Policy optimisation** – Fine-tune a copy of the conversational LLM with reinforcement learning (e.g., PPO, DPO, or RRHF) so that it maximises the learned reward function while maintaining helpful language alignment.
6. **Evaluation** – Run out-of-sample backtests and live paper trading with guardrails enabled to measure whether the RLHF-aligned policy improves Sharpe, win rate, or max drawdown compared to the baseline agent.

## Data requirements

* **Historical coverage** – Preferably several hundred independent trade scenarios. Thin datasets can be augmented with synthetic perturbations, but genuine market sequences provide the most reliable gradients.
* **Feature parity** – Ensure that the features supplied to the LLM during fine-tuning match those available at inference time (news feeds, sentiment scores, volatility regime labels, etc.).
* **Label hygiene** – Normalise P&L outcomes by volatility or position size to avoid over-weighting high-beta episodes. Record whether a trade was skipped for exogenous reasons (maintenance, API outage) to prevent misleading rewards.

## Implementation sketch

1. Extend the `trade_logger` pipeline to snapshot the full dialogue context, veto decisions, and realised performance for each trade candidate.
2. Build an offline labelling script that joins these logs with historical OHLCV data to compute outcome metrics (e.g., peak adverse excursion, final R multiple).
3. Train a lightweight reward model (could be a smaller transformer fine-tuned on dialogue embeddings) that predicts the desirability of an action sequence.
4. Use existing RLHF tooling (TRL, Accelerate) to fine-tune a copy of the base LLM against the reward model, optionally mixing in supervised data to stabilise training.
5. Deploy the RLHF-tuned model in shadow mode first, comparing its approvals with the production agent before promoting it to the main decision loop.

## Risks and mitigations

* **Small sample sizes** – Limited trade history may cause overfitting. Mitigate by applying cross-validation, regularisation, and conservative clipping of reward magnitudes.
* **Non-stationarity** – Market regimes shift; periodically refresh the reward model with new data and monitor for concept drift.
* **Operational safety** – Keep deterministic guardrails active. The RLHF policy should act as an advisor whose suggestions still pass through risk veto layers.

## Next steps

* Define the exact schema for captured dialogues and outcomes in `docs/trade_record_format.md`.
* Prototype the reward model training loop using a held-out set of 2023–2024 trades.
* Evaluate whether incremental RLHF updates deliver material improvements over simpler heuristic adjustments before committing to full-scale deployment.
