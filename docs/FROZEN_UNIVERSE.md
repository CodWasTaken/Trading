# Frozen major-stock universe

The initial governed universe is
`config/universes/us-liquid-large-cap-v1.json`: 48 U.S.-listed common stocks
across nine sectors, effective 2025-01-02.

The manifest freezes:

- ordered symbols and sector classifications;
- exchange and security type;
- effective date;
- inclusion rules;
- median daily dollar-volume threshold;
- minimum price;
- minimum historical coverage;
- shortability requirement;
- hashes of source declarations;
- its own canonical manifest hash.

The bundled source is an operator-frozen constituent and sector declaration,
not raw vendor liquidity evidence. Before treating an experiment as production
research, archive the immutable vendor screen, corporate-action history,
delistings, price/liquidity calculations, and borrow evidence used to verify
every inclusion rule. Update those source hashes in a new manifest; never edit a
manifest already referenced by an experiment.

## Experiment binding

Dataset construction, calibration training, sealed holdout splitting/evaluation,
replay, replay diagnostics, and paper startup accept a universe manifest and
require an exact symbol set. They retain the universe ID, effective date,
ordered symbols, manifest path, and manifest SHA-256. A missing symbol,
unexpected symbol, source hash mismatch, manifest hash mismatch, or date before
the effective date fails closed.

A historical experiment must select the manifest that was declared for that
experiment. Do not use this manifest for dates before its effective date, and do
not replace it with a later list of companies that are large today. A new
universe requires a new ID, effective date, evidence archive, and manifest hash.

The shortability rule does not make every symbol continuously borrowable. The
runtime still requires current easy-to-borrow status at every short-opening
decision, and missing status rejects the short.
