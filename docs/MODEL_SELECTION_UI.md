# Model catalogue and paper selection

The dashboard Models page reads `/v1/models/catalog` and shows registered versions, aliases, families, tasks, feature schemas, calibration metrics, holdout evidence, artifact verification, monitoring state and exact selection blockers.

`POST /v1/control/models/select` requires the configured control API key. The execution engine must be paused or the kill switch engaged. The selected model must be historically promoted, hash-verified, compatible with the frozen runtime universe and features, and not invalidated by monitoring.

A successful selection appends an immutable `active-paper` alias-history event and swaps the in-memory paper strategy. It does not promote a challenger, change sealed evidence or create live-money authority. The repository remains paper-only.
