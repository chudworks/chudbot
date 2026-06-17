import type { ReactNode } from 'react';
import type { CostAmount, ModelInfo, ToolTrace, TurnView, UsageRecord, UsageSubject } from '../types';

interface Props {
  turns: TurnView[];
  modelInfo: ModelInfo[];
}

type Metric = {
  seen: boolean;
  value: number;
};

type CostTotals = {
  usd: Metric;
  usdEstimated: boolean;
  native: Map<string, NativeCost>;
};

type NativeCost = {
  value: number;
  estimated: boolean;
};

type UsageGroup = {
  key: string;
  provider: string;
  model: string | null;
  subject: string;
  count: number;
  inputTokens: Metric;
  cachedTokens: Metric;
  outputTokens: Metric;
  reasoningTokens: Metric;
  totalTokens: Metric;
  costs: CostTotals;
};

type ContextUsage = {
  usedTokens: number;
  limitTokens: number;
};

const USD_TICKS_PER_DOLLAR = 10_000_000_000;

export default function UsageSummary({ turns, modelInfo }: Props) {
  const records = collectConversationUsage(turns);
  const contextUsage = currentContextUsage(turns, modelInfo);
  if (records.length === 0 && !contextUsage) return null;

  const summary = summarize(records);
  const groups = groupUsage(records);

  return (
    <section className="usage-summary" aria-label="Conversation usage">
      <div className="usage-summary__metrics">
        <UsageMetric label="Cost" value={formatCosts(summary.costs)} />
        {contextUsage && <UsageMetric label="Context" value={<ContextUsageValue usage={contextUsage} />} />}
        <UsageMetric label="Total tokens" value={formatMetric(summary.totalTokens)} />
        <UsageMetric label="Reasoning" value={formatMetric(summary.reasoningTokens)} />
        <UsageMetric label="Cached input" value={formatMetric(summary.cachedTokens)} />
      </div>
      {records.length > 0 && (
        <details className="usage-summary__details">
          <summary>Usage by source</summary>
          <div className="usage-summary__table-wrap">
            <table className="usage-table">
              <thead>
                <tr>
                  <th scope="col">Work</th>
                  <th scope="col">Provider/model</th>
                  <th scope="col" className="usage-table__num">Total</th>
                  <th scope="col" className="usage-table__num">Input</th>
                  <th scope="col" className="usage-table__num">Cached</th>
                  <th scope="col" className="usage-table__num">Output</th>
                  <th scope="col" className="usage-table__num">Reasoning</th>
                  <th scope="col" className="usage-table__num">Cost</th>
                </tr>
              </thead>
              <tbody>
                {groups.map((group) => (
                  <tr key={group.key}>
                    <td>
                      <span className="usage-table__subject">{group.subject}</span>
                      <span className="usage-table__count">
                        {formatNumber(group.count)} {group.count === 1 ? 'record' : 'records'}
                      </span>
                    </td>
                    <td>
                      <code>{providerModelLabel(group.provider, group.model)}</code>
                    </td>
                    <td className="usage-table__num">{formatMetric(group.totalTokens)}</td>
                    <td className="usage-table__num">{formatMetric(group.inputTokens)}</td>
                    <td className="usage-table__num">{formatMetric(group.cachedTokens)}</td>
                    <td className="usage-table__num">{formatMetric(group.outputTokens)}</td>
                    <td className="usage-table__num">{formatMetric(group.reasoningTokens)}</td>
                    <td className="usage-table__num">{formatCosts(group.costs)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </section>
  );
}

function UsageMetric({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="usage-summary__metric">
      <span className="usage-summary__metric-label">{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ContextUsageValue({ usage }: { usage: ContextUsage }) {
  return (
    <>
      {formatCompactTokens(usage.usedTokens)}
      <span className="usage-summary__context-separator"> / </span>
      {formatCompactTokens(usage.limitTokens)}
    </>
  );
}

function collectConversationUsage(turns: TurnView[]): UsageRecord[] {
  return turns.flatMap(collectTurnUsage);
}

function currentContextUsage(turns: TurnView[], modelInfo: ModelInfo[]): ContextUsage | null {
  const infoByModel = modelInfoLookup(modelInfo);
  for (let turnIndex = turns.length - 1; turnIndex >= 0; turnIndex -= 1) {
    const turn = turns[turnIndex];
    for (let usageIndex = turn.usage.length - 1; usageIndex >= 0; usageIndex -= 1) {
      const record = turn.usage[usageIndex];
      if (record.subject.kind !== 'model_step') continue;
      const inputTokens = finiteNumber(record.input_tokens);
      if (inputTokens == null) continue;
      const outputTokens = finiteNumber(record.output_tokens) ?? 0;
      const model = record.model ?? turn.turn.model;
      if (!model) continue;
      const info = infoByModel.get(modelInfoKey(record.provider, model));
      const limitTokens = finiteNumber(info?.context_window_tokens);
      if (limitTokens == null || limitTokens <= 0) return null;
      return {
        usedTokens: inputTokens + outputTokens,
        limitTokens,
      };
    }
  }
  return null;
}

function modelInfoLookup(modelInfo: ModelInfo[]): Map<string, ModelInfo> {
  const out = new Map<string, ModelInfo>();
  for (const info of modelInfo) {
    setModelInfo(out, info.provider, info.requested_model, info);
    setModelInfo(out, info.provider, info.model, info);
  }
  return out;
}

function setModelInfo(out: Map<string, ModelInfo>, provider: string, model: string, info: ModelInfo) {
  const key = modelInfoKey(provider, model);
  const existing = out.get(key);
  if (!existing || existing.context_window_tokens == null) {
    out.set(key, info);
  }
}

function modelInfoKey(provider: string, model: string): string {
  return `${provider}\u0000${model}`;
}

function collectTurnUsage(turn: TurnView): UsageRecord[] {
  const records = [...turn.usage];
  const persistedCounts = new Map<string, number>();
  for (const record of records) {
    const key = usageRecordKey(record);
    persistedCounts.set(key, (persistedCounts.get(key) ?? 0) + 1);
  }

  for (const traceRecord of turn.tool_trace.flatMap(usageFromTrace)) {
    const key = usageRecordKey(traceRecord);
    const persistedCount = persistedCounts.get(key) ?? 0;
    if (persistedCount > 0) {
      persistedCounts.set(key, persistedCount - 1);
    } else {
      records.push(traceRecord);
    }
  }

  return records;
}

function usageFromTrace(trace: ToolTrace): UsageRecord[] {
  switch (trace.kind) {
    case 'client':
      return trace.trace.usage;
    case 'server':
      return trace.tool.usage;
    case 'grounding':
      return [];
  }
}

function summarize(records: UsageRecord[]): UsageGroup {
  const group = emptyGroup('summary', '', null, '');
  for (const record of records) {
    addRecord(group, record);
  }
  return group;
}

function groupUsage(records: UsageRecord[]): UsageGroup[] {
  const groups = new Map<string, UsageGroup>();
  for (const record of records) {
    const subject = subjectLabel(record.subject);
    const model = record.model ?? null;
    const key = `${record.provider}\u0000${model ?? ''}\u0000${subject}`;
    let group = groups.get(key);
    if (!group) {
      group = emptyGroup(key, record.provider, model, subject);
      groups.set(key, group);
    }
    addRecord(group, record);
  }
  return [...groups.values()].sort((a, b) => {
    const costSort = b.costs.usd.value - a.costs.usd.value;
    if (costSort !== 0) return costSort;
    const tokenSort = b.totalTokens.value - a.totalTokens.value;
    if (tokenSort !== 0) return tokenSort;
    return a.subject.localeCompare(b.subject);
  });
}

function emptyGroup(key: string, provider: string, model: string | null, subject: string): UsageGroup {
  return {
    key,
    provider,
    model,
    subject,
    count: 0,
    inputTokens: emptyMetric(),
    cachedTokens: emptyMetric(),
    outputTokens: emptyMetric(),
    reasoningTokens: emptyMetric(),
    totalTokens: emptyMetric(),
    costs: emptyCosts(),
  };
}

function addRecord(group: UsageGroup, record: UsageRecord) {
  group.count += 1;
  addMetric(group.inputTokens, record.input_tokens);
  addMetric(group.cachedTokens, record.cached_input_tokens);
  addMetric(group.outputTokens, record.output_tokens);
  addMetric(group.reasoningTokens, record.reasoning_tokens);
  addMetric(group.totalTokens, record.total_tokens);
  addCost(group.costs, record.cost);
}

function emptyMetric(): Metric {
  return { seen: false, value: 0 };
}

function addMetric(metric: Metric, value: number | null | undefined) {
  const numeric = finiteNumber(value);
  if (numeric == null) return;
  metric.seen = true;
  metric.value += numeric;
}

function finiteNumber(value: number | null | undefined): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function emptyCosts(): CostTotals {
  return {
    usd: emptyMetric(),
    usdEstimated: false,
    native: new Map(),
  };
}

function addCost(totals: CostTotals, cost: CostAmount | null | undefined) {
  if (!cost) return;
  const amount = Number(cost.amount);
  if (!Number.isFinite(amount)) return;

  if (cost.unit === 'usd_ticks') {
    addMetric(totals.usd, amount / USD_TICKS_PER_DOLLAR);
    totals.usdEstimated ||= cost.estimated;
    return;
  }
  if (cost.unit === 'usd') {
    addMetric(totals.usd, amount);
    totals.usdEstimated ||= cost.estimated;
    return;
  }

  const native = totals.native.get(cost.unit) ?? { value: 0, estimated: false };
  native.value += amount;
  native.estimated ||= cost.estimated;
  totals.native.set(cost.unit, native);
}

function formatCosts(costs: CostTotals): string {
  const parts: string[] = [];
  if (costs.usd.seen) {
    parts.push(formatUsd(costs.usd.value, costs.usdEstimated));
  }
  for (const [unit, native] of costs.native) {
    parts.push(`${formatNumber(native.value)} ${unit}${native.estimated ? ' est.' : ''}`);
  }
  return parts.length > 0 ? parts.join(', ') : 'n/a';
}

function formatUsd(amount: number, estimated: boolean): string {
  const digits = amount > 0 && amount < 1 ? 4 : 2;
  const formatted = new Intl.NumberFormat(undefined, {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(amount);
  return estimated ? `${formatted} est.` : formatted;
}

function formatMetric(metric: Metric): string {
  return metric.seen ? formatNumber(metric.value) : 'n/a';
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat().format(value);
}

function formatCompactTokens(value: number): string {
  const abs = Math.abs(value);
  if (abs < 1000) return formatNumber(Math.round(value));
  if (abs < 999_950) return `${formatCompactDecimal(value / 1000)}k`;
  if (abs < 999_950_000) return `${formatCompactDecimal(value / 1_000_000)}M`;
  return `${formatCompactDecimal(value / 1_000_000_000)}B`;
}

function formatCompactDecimal(value: number): string {
  return new Intl.NumberFormat(undefined, {
    maximumFractionDigits: 1,
  }).format(value);
}

function providerModelLabel(provider: string, model: string | null): string {
  return model ? `${provider}/${model}` : provider;
}

function subjectLabel(subject: UsageSubject): string {
  const base = titleCase(subject.kind);
  return subject.name ? `${base}: ${subject.name}` : base;
}

function titleCase(value: string): string {
  return value
    .split('_')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function usageRecordKey(record: UsageRecord): string {
  return stableJson({
    provider: record.provider,
    model: record.model ?? null,
    subject: record.subject,
    input_tokens: record.input_tokens ?? null,
    cached_input_tokens: record.cached_input_tokens ?? null,
    output_tokens: record.output_tokens ?? null,
    reasoning_tokens: record.reasoning_tokens ?? null,
    total_tokens: record.total_tokens ?? null,
    cost: record.cost ?? null,
    raw: record.raw ?? null,
  });
}

function stableJson(value: unknown): string {
  if (value === null || typeof value !== 'object') {
    return JSON.stringify(value) ?? 'undefined';
  }
  if (Array.isArray(value)) {
    return `[${value.map(stableJson).join(',')}]`;
  }
  const entries = Object.entries(value as Record<string, unknown>).sort(([a], [b]) =>
    a.localeCompare(b)
  );
  return `{${entries
    .map(([key, entry]) => `${JSON.stringify(key)}:${stableJson(entry)}`)
    .join(',')}}`;
}
