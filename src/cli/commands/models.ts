import { readModelsSource, writeModelsProvider } from "../../features/models/models-source-io.js";
import type { RulesyncModelProvider } from "../../features/models/rulesync-models.js";
// `models fetch`: keep `.rulesync/models.jsonc` populated without
// hand-copying. Fetch copies one provider block from models.dev so `generate`
// stays offline and repeatable. Edits preserve the source file's comments.
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
