import {
  mergeDiscoveredModels,
  readModelsSource,
  writeModelsProvider,
} from "../../features/models/models-source-io.js";
import type {
  RulesyncModel,
  RulesyncModelProvider,
} from "../../features/models/rulesync-models.js";
// `models fetch` / `models discover`: keep `.rulesync/models.jsonc`
// populated without hand-copying. Fetch copies one provider block from
// models.dev so `generate` stays offline and repeatable; discover fills model
// ids (and server-reported limits only) from a live OpenAI-compatible
// endpoint. Both edit the source file in place, preserving comments.
import type { Logger } from "../../utils/logger.js";
import { isRecord } from "../../utils/type-guards.js";

const MODELS_DEV_API_URL = "https://models.dev/api.json";

type FetchFn = typeof fetch;

async function fetchModelsDevCatalog(fetchFn: FetchFn): Promise<Record<string, unknown>> {
  const response = await fetchFn(MODELS_DEV_API_URL);
  if (!response.ok) {
    throw new Error(`models.dev request failed with status ${response.status}.`);
  }
  const json: unknown = await response.json();
  if (typeof json !== "object" || json === null) {
    throw new Error("models.dev returned an unexpected response shape.");
  }
  return json as Record<string, unknown>;
}

export async function modelsFetchCommand(
  logger: Logger,
  options: { provider: string; outputRoot?: string; fetchFn?: FetchFn },
): Promise<void> {
  const source = await readModelsSource({ outputRoot: options.outputRoot });
  if (!source) {
    throw new Error("No .rulesync/models.jsonc found. Run with a models source file present.");
  }
  const catalog = await fetchModelsDevCatalog(options.fetchFn ?? fetch);
  if (!isRecord(catalog.providers)) {
    throw new Error("models.dev returned a catalog without providers.");
  }
  const block = catalog.providers[options.provider];
  if (!isRecord(block)) {
    throw new Error(`models.dev has no provider "${options.provider}".`);
  }
  const hadExisting = isRecord(source.json.providers)
    ? isRecord(source.json.providers[options.provider])
    : false;
  await writeModelsProvider({
    source,
    providerId: options.provider,
    provider: block as unknown as RulesyncModelProvider,
  });
  logger.success(
    `${hadExisting ? "Updated" : "Added"} provider "${options.provider}" from models.dev.`,
  );
}
type DiscoverResponse = {
  data?: { id?: unknown } & Record<string, unknown>[];
};

/** Context fields servers already attach to `GET /v1/models` entries. */
function discoverContextLimit(entry: Record<string, unknown>): number | undefined {
  for (const key of ["max_model_len", "context_length", "max_context_length"]) {
    if (typeof entry[key] === "number") return entry[key] as number;
  }
  return undefined;
}

export async function modelsDiscoverCommand(
  logger: Logger,
  options: { provider: string; outputRoot?: string; fetchFn?: FetchFn },
): Promise<void> {
  const source = await readModelsSource({ outputRoot: options.outputRoot });
  if (!source) {
    throw new Error("No .rulesync/models.jsonc found. Run with a models source file present.");
  }
  const existing = isRecord(source.json.providers)
    ? source.json.providers[options.provider]
    : undefined;
  if (!isRecord(existing)) {
    throw new Error(`No provider "${options.provider}" in .rulesync/models.jsonc.`);
  }
  const api = typeof existing.api === "string" ? existing.api : undefined;
  if (!api) {
    throw new Error(`Provider "${options.provider}" names no api base URL to discover from.`);
  }
  const envNames = Array.isArray(existing.env) ? existing.env : [];
  const apiKey = typeof envNames[0] === "string" ? process.env[envNames[0]] : undefined;

  const url = `${api.replace(/\/+$/, "")}/models`;
  const response = await (options.fetchFn ?? fetch)(url, {
    headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : {},
  });
  if (!response.ok) {
    throw new Error(`Discovery request to ${url} failed with status ${response.status}.`);
  }
  const json = (await response.json()) as DiscoverResponse;
  if (!Array.isArray(json.data)) {
    throw new Error(`Discovery response from ${url} has no data array.`);
  }

  const incoming: Record<string, RulesyncModel> = {};
  for (const entry of json.data) {
    if (typeof entry !== "object" || entry === null || typeof entry.id !== "string") continue;
    const context = discoverContextLimit(entry as Record<string, unknown>);
    incoming[entry.id] = {
      id: entry.id,
      ...(context !== undefined && { limit: { context } }),
    };
  }
  const existingRecord = existing as Record<string, unknown>;
  const before = Object.keys(isRecord(existingRecord.models) ? existingRecord.models : {}).length;
  const merged = mergeDiscoveredModels({
    incoming: incoming as unknown as Record<string, unknown>,
    existing: existingRecord.models,
  });
  await writeModelsProvider({
    source,
    providerId: options.provider,
    provider: {
      ...(existingRecord as unknown as RulesyncModelProvider),
      models: merged as unknown as RulesyncModelProvider["models"],
    },
  });
  const added = Object.keys(merged).length - before;
  logger.success(
    `Discovered ${json.data.length} models from "${options.provider}" (${added} new).`,
  );
}
