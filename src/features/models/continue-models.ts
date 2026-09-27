// Continue mapping for the `models` feature.
//
// Each model becomes an entry of the `models:` list in `config.yaml`
// (`provider: openai` with `apiBase` for OpenAI-compatible endpoints).
// Deliberate omissions: API keys stay in Continue's secrets handling (never
// written here), capabilities are additive-only in Continue so a negative is
// inexpressible and dropped, and the default model is UI state with no config
// key. Entry names are `<provider>/<model>` so import splits them back
// unambiguously on the first `/`.
import { join } from "node:path";

import { CONTINUE_DIR } from "../../constants/continue-paths.js";
import { ValidationResult } from "../../types/ai-file.js";
import { readFileContentOrNull } from "../../utils/file.js";
import { isPrototypePollutionKey } from "../../utils/prototype-pollution.js";
import { isRecord } from "../../utils/type-guards.js";
import {
  applySharedConfigPatch,
  parseSharedConfig,
  sharedConfigFileKey,
} from "../shared/shared-config-gateway.js";
import {
  forTargetModelConfig,
  type RulesyncModel,
  type RulesyncModelProvider,
} from "./rulesync-models.js";
import { RulesyncModels } from "./rulesync-models.js";
import {
  ToolModels,
  ToolModelsForDeletionParams,
  ToolModelsFromFileParams,
  ToolModelsFromRulesyncModelsParams,
  ToolModelsParams,
  ToolModelsSettablePaths,
} from "./tool-models.js";

const CONTINUE_CONFIG_FILE_NAME = "config.yaml";

function toContinueCapabilities(model: RulesyncModel): string[] | undefined {
  const capabilities: string[] = [];
  if (model.tool_call === true) capabilities.push("tool_use");
  if (model.modalities?.input?.includes("image") === true) capabilities.push("image_input");
  return capabilities.length > 0 ? capabilities : undefined;
}

function toContinueEntry(
  providerId: string,
  modelId: string,
  model: RulesyncModel,
  api: string | undefined,
): Record<string, unknown> {
  return {
    name: `${providerId}/${modelId}`,
    provider: "openai",
    model: modelId,
    ...(api !== undefined && { apiBase: api }),
    ...(model.limit?.context !== undefined && { contextLength: model.limit.context }),
    ...(model.limit?.output !== undefined && {
      defaultCompletionOptions: { maxTokens: model.limit.output },
    }),
    ...(toContinueCapabilities(model) !== undefined && {
      capabilities: toContinueCapabilities(model),
    }),
  };
}

function fromContinueEntry(entry: Record<string, unknown>): {
  providerId: string;
  modelId: string;
  model: RulesyncModel;
} | null {
  const name = typeof entry.name === "string" ? entry.name : undefined;
  const modelId = typeof entry.model === "string" ? entry.model : undefined;
  if (!name || !modelId) return null;
  // The generated name is `<provider>/<model>`; anything else cannot be
  // split back unambiguously and is skipped.
  const slash = name.indexOf("/");
  if (slash <= 0 || isPrototypePollutionKey(name.slice(0, slash))) return null;
  const providerId = name.slice(0, slash);
  if (name.slice(slash + 1) !== modelId || isPrototypePollutionKey(modelId)) return null;
  const completionOptions = (entry.defaultCompletionOptions as Record<string, unknown> | undefined)
    ?.maxTokens;
  const capabilities = Array.isArray(entry.capabilities)
    ? entry.capabilities.filter((c): c is string => typeof c === "string")
    : [];
  const model: RulesyncModel = { id: modelId };
  if (typeof entry.contextLength === "number") {
    model.limit = {
      context: entry.contextLength,
      ...(typeof completionOptions === "number" && { output: completionOptions }),
    };
  } else if (typeof completionOptions === "number") {
    model.limit = { output: completionOptions };
  }
  if (capabilities.includes("tool_use")) model.tool_call = true;
  if (capabilities.includes("image_input")) model.modalities = { input: ["image"] };
  return { providerId, modelId, model };
}

export class ContinueModels extends ToolModels {
  private config: Record<string, unknown>;

  constructor(params: ToolModelsParams) {
    super(params);
    this.config =
      this.fileContent !== undefined
        ? parseSharedConfig({ format: "yaml", fileContent: this.fileContent })
        : {};
  }

  getConfig(): Record<string, unknown> {
    return this.config;
  }

  /** config.yaml holds the whole Continue setup, so it is never deleted. */
  override isDeletable(): boolean {
    return false;
  }

  static getSettablePaths(_options: { global?: boolean } = {}): ToolModelsSettablePaths {
    // Same relative directory at both scopes; global mode only changes the
    // output root to the home directory.
    return { relativeDirPath: CONTINUE_DIR, relativeFilePath: CONTINUE_CONFIG_FILE_NAME };
  }

  static async fromFile({
    outputRoot = process.cwd(),
    validate = true,
    global = false,
  }: ToolModelsFromFileParams): Promise<ContinueModels> {
    const paths = this.getSettablePaths({ global });
    const fileContent =
      (await readFileContentOrNull(
        join(outputRoot, paths.relativeDirPath, paths.relativeFilePath),
      )) ?? "";
    return new ContinueModels({
      outputRoot,
      relativeDirPath: paths.relativeDirPath,
      relativeFilePath: paths.relativeFilePath,
      fileContent,
      validate,
      global,
    });
  }

  static async fromRulesyncModels({
    outputRoot = process.cwd(),
    rulesyncModels,
    validate = true,
    global = false,
    logger,
  }: ToolModelsFromRulesyncModelsParams): Promise<ContinueModels> {
    const paths = this.getSettablePaths({ global });
    const fileContent =
      (await readFileContentOrNull(
        join(outputRoot, paths.relativeDirPath, paths.relativeFilePath),
      )) ?? "";
    const { providers } = forTargetModelConfig({
      config: rulesyncModels.getJson(),
      toolTarget: "continue",
    });

    const entries: Record<string, unknown>[] = [];
    for (const [id, provider] of Object.entries(providers)) {
      if (isPrototypePollutionKey(id)) continue;
      const modelEntries = Object.entries(provider.models ?? {});
      if (modelEntries.length === 0) {
        logger?.warn(`Continue models: provider "${id}" names no models, skipping it.`);
        continue;
      }
      for (const [modelId, model] of modelEntries) {
        if (isPrototypePollutionKey(modelId)) continue;
        entries.push(toContinueEntry(id, modelId, model, provider.api));
      }
    }

    return new ContinueModels({
      outputRoot,
      relativeDirPath: paths.relativeDirPath,
      relativeFilePath: paths.relativeFilePath,
      fileContent: applySharedConfigPatch({
        fileKey: sharedConfigFileKey(paths),
        feature: "models",
        existingContent: fileContent,
        patch: { models: entries },
        filePath: join(outputRoot, paths.relativeDirPath, paths.relativeFilePath),
      }),
      validate,
      global,
    });
  }

  toRulesyncModels(): RulesyncModels {
    const providers: Record<string, RulesyncModelProvider> = {};
    const models = this.config.models;
    if (Array.isArray(models)) {
      for (const item of models) {
        if (!isRecord(item)) continue;
        const converted = fromContinueEntry(item);
        if (!converted) continue;
        const { providerId, modelId, model } = converted;
        const api = typeof item.apiBase === "string" ? item.apiBase : undefined;
        const existing: RulesyncModelProvider = providers[providerId] ?? { models: {} };
        if (api !== undefined && existing.api === undefined) existing.api = api;
        existing.models = { ...existing.models, [modelId]: model };
        providers[providerId] = existing;
      }
    }
    return this.toRulesyncModelsDefault({
      fileContent: JSON.stringify({ providers }, null, 2),
    });
  }

  validate(): ValidationResult {
    return { success: true, error: null };
  }

  static forDeletion({
    outputRoot = process.cwd(),
    relativeDirPath,
    relativeFilePath,
    global = false,
  }: ToolModelsForDeletionParams): ContinueModels {
    return new ContinueModels({
      outputRoot,
      relativeDirPath,
      relativeFilePath,
      fileContent: "",
      validate: false,
      global,
    });
  }
}
