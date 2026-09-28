// Read-modify-write for the `.rulesync/models.jsonc` source file itself,
// used by `models fetch` and `models discover`. Edits go through
// jsonc-parser's `modify` so comments and formatting in the hand-maintained
// file survive; a file that cannot be parsed fails closed.
import { join } from "node:path";

import { applyEdits, modify, parse as parseJsonc } from "jsonc-parser";

import {
  RULESYNC_MODELS_FILE_NAME,
  RULESYNC_RELATIVE_DIR_PATH,
} from "../../constants/rulesync-paths.js";
import { fileExistsStrict, readFileContent, writeFileContent } from "../../utils/file.js";
import { isRecord } from "../../utils/type-guards.js";
import type { RulesyncModelProvider } from "./rulesync-models.js";

export type ModelsSource = {
  filePath: string;
  content: string;
  json: Record<string, unknown>;
};

export async function readModelsSource({
  outputRoot = process.cwd(),
}: {
  outputRoot?: string;
} = {}): Promise<ModelsSource | null> {
  const filePath = join(outputRoot, RULESYNC_RELATIVE_DIR_PATH, RULESYNC_MODELS_FILE_NAME);
  if (!(await fileExistsStrict(filePath))) {
    return null;
  }
  const content = await readFileContent(filePath);
  const json = parseJsonc(content);
  if (typeof json !== "object" || json === null || Array.isArray(json)) {
    throw new Error(`Invalid models source file '${filePath}': expected a JSON object.`);
  }
  return { filePath, content, json: json as Record<string, unknown> };
}

/**
 * Fill each on-disk model with the incoming fields it lacks (notably
 * `limit`), without touching anything the user already set. New ids are
 * added; nothing is ever removed.
 */
export function mergeDiscoveredModels({
  incoming,
  existing,
}: {
  incoming: Record<string, unknown>;
  existing: unknown;
}): Record<string, unknown> {
  const base = isRecord(existing) ? existing : {};
  const merged: Record<string, unknown> = { ...base };
  for (const [id, model] of Object.entries(incoming)) {
    if (!isRecord(model)) continue;
    const current = isRecord(base[id]) ? base[id] : {};
    const incomingLimit = isRecord(model.limit) ? model.limit : {};
    const currentLimit = isRecord(current.limit) ? current.limit : {};
    merged[id] = {
      ...model,
      ...current,
      ...((Object.keys(incomingLimit).length > 0 || Object.keys(currentLimit).length > 0) && {
        limit: { ...incomingLimit, ...currentLimit },
      }),
    };
  }
  return merged;
}

/**
 * Set one provider block, preserving everything else byte-for-byte.
 * `mergeModels` unions the incoming models with the on-disk ones (existing
 * values win, missing limits are filled) instead of replacing them wholesale.
 */
export async function writeModelsProvider({
  source,
  providerId,
  provider,
  mergeModels = false,
}: {
  source: ModelsSource;
  providerId: string;
  provider: RulesyncModelProvider;
  mergeModels?: boolean;
}): Promise<void> {
  let block: unknown = provider;
  if (mergeModels) {
    const existing = isRecord(source.json.providers)
      ? source.json.providers[providerId]
      : undefined;
    block = {
      ...provider,
      models: mergeDiscoveredModels({
        incoming: (provider.models ?? {}) as Record<string, unknown>,
        existing: isRecord(existing) ? existing.models : undefined,
      }),
    };
  }
  const edits = modify(source.content, ["providers", providerId], block, {
    formattingOptions: { insertSpaces: true, tabSize: 2 },
  });
  await writeFileContent(source.filePath, applyEdits(source.content, edits));
}
