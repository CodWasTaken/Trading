"use client";

import { useCallback, useEffect, useState } from "react";
import styles from "./models.module.css";

type Capabilities = {
  family: string;
  implementation: string;
  task: string;
  feature_names: string[];
  forecast_horizon: number;
  required_history: number;
};

type CatalogueModel = {
  version: string;
  created_at: string;
  aliases: string[];
  capabilities: Capabilities | null;
  artifact_verified: boolean;
  metrics: Record<string, unknown>;
  holdout: Record<string, unknown> | null;
  monitoring_status: string;
  eligible_for_active_paper: boolean;
  selection_blockers: string[];
};

type Catalogue = {
  execution_scope: "paper_only";
  live_money_authorized: false;
  aliases: Record<string, string>;
  models: CatalogueModel[];
};

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const CONTROL_KEY = process.env.NEXT_PUBLIC_CONTROL_API_KEY;

function metric(model: CatalogueModel, key: string): string {
  const value = model.metrics[key];
  return typeof value === "number" ? value.toFixed(4) : "—";
}

export default function ModelsPage() {
  const [catalogue, setCatalogue] = useState<Catalogue | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reasons, setReasons] = useState<Record<string, string>>({});

  const refresh = useCallback(async () => {
    try {
      const response = await fetch(`${API}/v1/models/catalog`, { cache: "no-store" });
      if (!response.ok) throw new Error(`API returned ${response.status}`);
      setCatalogue(await response.json());
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to load model catalogue");
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  async function selectModel(version: string) {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (CONTROL_KEY) headers["X-Trading-API-Key"] = CONTROL_KEY;
    const response = await fetch(`${API}/v1/control/models/select`, {
      method: "POST",
      headers,
      body: JSON.stringify({ version, reason: reasons[version] ?? "operator web selection" }),
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      setError(payload.detail ?? `Selection failed with ${response.status}`);
      return;
    }
    await refresh();
  }

  return (
    <main className={styles.shell}>
      <header className={styles.header}>
        <div>
          <h1>Model catalogue</h1>
          <p>
            Compare registered challengers, historical champions, holdout evidence,
            monitoring health, and the model currently selected for paper execution.
          </p>
        </div>
      </header>
      <div className={styles.notice}>
        Selection changes the audited <code>active-paper</code> alias only. The engine
        must be paused or kill-switched. This application cannot authorize live money.
      </div>
      {error && <div className={`${styles.notice} ${styles.error}`}>{error}</div>}
      {!catalogue && !error && <p>Loading governed registry…</p>}
      <section className={styles.grid}>
        {catalogue?.models.map((model) => (
          <article className={styles.card} key={model.version}>
            <div className={styles.cardHeader}>
              <h2>{model.version}</h2>
              <span className={`${styles.badge} ${model.artifact_verified ? styles.good : styles.bad}`}>
                {model.artifact_verified ? "hash verified" : "artifact invalid"}
              </span>
            </div>
            <div className={styles.badges}>
              {model.aliases.map((alias) => <span className={styles.badge} key={alias}>{alias}</span>)}
              <span className={styles.badge}>{model.capabilities?.family ?? "legacy"}</span>
              <span className={styles.badge}>{model.capabilities?.implementation ?? "ridge"}</span>
            </div>
            <dl className={styles.facts}>
              <dt>Task</dt><dd>{model.capabilities?.task ?? "legacy return"}</dd>
              <dt>Horizon</dt><dd>{model.capabilities?.forecast_horizon ?? "—"} bars</dd>
              <dt>History</dt><dd>{model.capabilities?.required_history ?? "—"} observations</dd>
              <dt>Net return</dt><dd>{metric(model, "net_return")}</dd>
              <dt>Excess return</dt><dd>{metric(model, "benchmark_excess_return")}</dd>
              <dt>Sharpe</dt><dd>{metric(model, "sharpe")}</dd>
              <dt>Drawdown</dt><dd>{metric(model, "max_drawdown")}</dd>
              <dt>Monitoring</dt><dd>{model.monitoring_status}</dd>
            </dl>
            <p className={styles.blockers}>
              {model.selection_blockers.length
                ? model.selection_blockers.join(" · ")
                : "Eligible after the engine is safely paused."}
            </p>
            <div className={styles.actions}>
              <input
                aria-label={`Reason for selecting ${model.version}`}
                placeholder="Selection reason"
                value={reasons[model.version] ?? ""}
                onChange={(event) => setReasons({ ...reasons, [model.version]: event.target.value })}
              />
              <button
                disabled={!model.eligible_for_active_paper}
                onClick={() => void selectModel(model.version)}
              >
                Select
              </button>
            </div>
          </article>
        ))}
      </section>
    </main>
  );
}
