"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

type Position = {
  symbol: string;
  quantity: number;
  average_price: number;
  last_price: number;
  market_value: number;
  unrealized_pnl: number;
};

type Portfolio = {
  cash: number;
  equity: number;
  gross_exposure: number;
  daily_pnl: number;
  drawdown: number;
  positions: Position[];
  trades_today: number;
};

type LedgerItem = Record<string, unknown>;
type Summary = {
  portfolio: Portfolio;
  kill_switch: boolean;
  engine_running: boolean;
  active_strategy: string;
  symbols: string[];
  recent: {
    news: LedgerItem[];
    proposals: LedgerItem[];
    decisions: LedgerItem[];
    orders: LedgerItem[];
    fills: LedgerItem[];
  };
};

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const money = new Intl.NumberFormat("en-PL", {
  style: "currency",
  currency: "USD",
});
const percent = new Intl.NumberFormat("en-PL", {
  style: "percent",
  maximumFractionDigits: 2,
});

function value(item: LedgerItem, key: string): string {
  const result = item[key];
  return result === undefined || result === null ? "—" : String(result);
}

export default function Dashboard() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const response = await fetch(`${API}/v1/dashboard/summary`, {
        cache: "no-store",
      });
      if (!response.ok) throw new Error(`API returned ${response.status}`);
      setSummary(await response.json());
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Unable to load API");
    }
  }, []);

  useEffect(() => {
    void refresh();
    const interval = window.setInterval(refresh, 5000);
    const socketUrl = API.replace(/^http/, "ws") + "/v1/live/events";
    const socket = new WebSocket(socketUrl);
    socket.onopen = () => setConnected(true);
    socket.onclose = () => setConnected(false);
    socket.onerror = () => setConnected(false);
    socket.onmessage = () => void refresh();
    return () => {
      window.clearInterval(interval);
      socket.close();
    };
  }, [refresh]);

  const decisions = useMemo(
    () => summary?.recent.decisions.slice(0, 12) ?? [],
    [summary],
  );

  async function setKillSwitch(enabled: boolean) {
    await fetch(`${API}/v1/control/kill-switch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    });
    await refresh();
  }

  if (!summary) {
    return (
      <main className="shell">
        <div className="empty">
          {error ?? "Connecting to trading engine…"}
        </div>
      </main>
    );
  }

  const portfolio = summary.portfolio;
  return (
    <main className="shell">
      <header className="header">
        <div>
          <p className="eyebrow">Paper trading · control room</p>
          <h1>AI Trading Dashboard</h1>
          <p className="muted">
            Explainable signals. Deterministic risk. Every action audited.
          </p>
        </div>
        <div className="headerActions">
          <span className={`status ${connected ? "good" : "bad"}`}>
            {connected ? "Live" : "Disconnected"}
          </span>
          <button
            className={summary.kill_switch ? "danger active" : "danger"}
            onClick={() => void setKillSwitch(!summary.kill_switch)}
          >
            {summary.kill_switch
              ? "Release kill switch"
              : "Engage kill switch"}
          </button>
        </div>
      </header>

      {error && <div className="alert">{error}</div>}

      <section className="metrics">
        <Metric label="Portfolio equity" value={money.format(portfolio.equity)} />
        <Metric label="Today’s P&L" value={money.format(portfolio.daily_pnl)} />
        <Metric
          label="Gross exposure"
          value={money.format(portfolio.gross_exposure)}
        />
        <Metric label="Drawdown" value={percent.format(portfolio.drawdown)} />
        <Metric label="Cash" value={money.format(portfolio.cash)} />
        <Metric label="Trades today" value={String(portfolio.trades_today)} />
      </section>

      <section className="grid">
        <Panel
          title="Live AI decisions"
          subtitle="Newest risk checks first"
          wide
        >
          <div className="tableWrap">
            <table>
              <thead>
                <tr>
                  <th>Status</th>
                  <th>Proposal</th>
                  <th>Approved</th>
                  <th>Reasons</th>
                  <th>Time</th>
                </tr>
              </thead>
              <tbody>
                {decisions.map((item, index) => (
                  <tr key={`${value(item, "proposal_id")}-${index}`}>
                    <td>
                      <span className={`pill ${value(item, "status")}`}>
                        {value(item, "status")}
                      </span>
                    </td>
                    <td className="mono">
                      {value(item, "proposal_id").slice(0, 8)}
                    </td>
                    <td>
                      {money.format(Number(item.approved_notional ?? 0))}
                    </td>
                    <td>
                      {Array.isArray(item.reasons)
                        ? item.reasons.join(", ")
                        : "—"}
                    </td>
                    <td>
                      {new Date(
                        value(item, "checked_at"),
                      ).toLocaleTimeString("en-PL")}
                    </td>
                  </tr>
                ))}
                {!decisions.length && (
                  <tr>
                    <td colSpan={5} className="muted">
                      Waiting for enough market observations…
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </Panel>

        <Panel
          title="Open positions"
          subtitle={`${portfolio.positions.length} active`}
        >
          <div className="stack">
            {portfolio.positions.map((position) => (
              <div className="position" key={position.symbol}>
                <div>
                  <strong>{position.symbol}</strong>
                  <span>{position.quantity.toFixed(3)} shares</span>
                </div>
                <div className="right">
                  <strong>{money.format(position.market_value)}</strong>
                  <span>{money.format(position.unrealized_pnl)}</span>
                </div>
              </div>
            ))}
            {!portfolio.positions.length && (
              <p className="muted">No open positions.</p>
            )}
          </div>
        </Panel>

        <Panel title="Breaking news" subtitle="Ticker-linked event stream">
          <div className="stack">
            {summary.recent.news.slice(0, 8).map((item, index) => (
              <article
                className="news"
                key={`${value(item, "id")}-${index}`}
              >
                <div>
                  <strong>{value(item, "symbol")}</strong>
                  <span>{value(item, "event_type")}</span>
                </div>
                <p>{value(item, "headline")}</p>
                <small>
                  {value(item, "source")} · sentiment{" "}
                  {Number(item.sentiment ?? 0).toFixed(2)}
                </small>
              </article>
            ))}
            {!summary.recent.news.length && (
              <p className="muted">News events will appear here.</p>
            )}
          </div>
        </Panel>

        <Panel title="System state" subtitle="Fail-closed controls">
          <dl className="facts">
            <div>
              <dt>Engine</dt>
              <dd>{summary.engine_running ? "running" : "paused"}</dd>
            </div>
            <div>
              <dt>Strategy</dt>
              <dd>{summary.active_strategy}</dd>
            </div>
            <div>
              <dt>Kill switch</dt>
              <dd>{summary.kill_switch ? "ENGAGED" : "clear"}</dd>
            </div>
            <div>
              <dt>Universe</dt>
              <dd>{summary.symbols.join(", ")}</dd>
            </div>
            <div>
              <dt>Execution</dt>
              <dd>paper only</dd>
            </div>
          </dl>
        </Panel>
      </section>
    </main>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Panel({
  title,
  subtitle,
  wide = false,
  children,
}: {
  title: string;
  subtitle: string;
  wide?: boolean;
  children: React.ReactNode;
}) {
  return (
    <section className={`panel ${wide ? "wide" : ""}`}>
      <div className="panelHead">
        <div>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
      </div>
      {children}
    </section>
  );
}
