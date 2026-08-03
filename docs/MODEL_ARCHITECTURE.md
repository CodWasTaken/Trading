# Governed multi-model architecture

Version 0.23 introduces a common `TradingModel` contract around the existing ridge baseline and new elastic-net, LightGBM, XGBoost, CatBoost, MLP, LSTM, GRU, TCN, PatchTST and iTransformer adapters.

Every heterogeneous model artifact records its family, implementation, task, feature schema, forecast horizon, required history, deterministic seed, hyperparameters, dependency versions, provenance and SHA-256. Loading fails closed when the artifact hash or capability manifest differs from the registry.

Optional libraries are imported lazily. The base API, dashboard and runtime smoke do not install GPU frameworks or tree libraries. Install `.[models-tabular]`, `.[models-sequence]` or `.[models]` explicitly.

All training accepts calibration-role datasets only and uses nested purged chronological folds. A five-bar target remains a historical label, never a live feature. New models register as challengers; they still require the existing sealed holdout and promotion gates.

Only live-compatible return-regression models using runtime features and at most 20 observations of history can be selected for paper execution. Forecasting, language and portfolio-policy models must first produce governed features or a compatible return model. No model type bypasses risk, cost, replay, holdout or paper-graduation controls.
