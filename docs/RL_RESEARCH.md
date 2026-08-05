# Reinforcement-learning research

RL policies are calibration-only challengers. They do not bypass the signed portfolio, conservative cost model, frozen universe, sealed holdout, or paper-only boundary.

## Environment

`PortfolioResearchEnvironment` exposes current point-in-time features and previous positions. The next five-bar realized return is consumed only after an action to calculate reward. Actions are normalized to `[-1, 1]`, mapped into configured long and short position limits, and rescaled to gross-exposure limits.

The reward starts with return after spread, slippage, commissions, regulatory fees, market-impact stress, borrow costs, dividend replacement, and margin interest. It then applies configurable penalties for drawdown, turnover, concentration, gross exposure, borrow dependence, and constraint violations.

## Algorithms

The optional Stable-Baselines3 adapter supports PPO, A2C, SAC, and TD3. Training requires a dataset whose role is `calibration` and which is not sealed.

```bash
trading-integrations rl-train \
  --dataset .trading/datasets/cycle-01-calibration.jsonl \
  --spec config/rl/ppo-conservative-v1.json \
  --cost-config config/costs/conservative-us-paper-v1.json \
  --output .trading/rl/cycle-01-ppo
```

The resulting policy manifest is hash-bound to the dataset, metadata, cost model, algorithm configuration, symbols, and policy artifact. RL policies are recorded as `portfolio_policy` and `live_compatible=false` until a future shared-engine action adapter and deterministic replay prove compatibility.

## Evaluation

Use walk-forward episodes and multiple seeds. Compare against cash, equal-weight, ridge, tree models, and simple deterministic allocation policies. A high training reward is not promotion evidence. Only three frozen finalists may be evaluated on the sealed holdout.
