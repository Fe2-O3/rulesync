// Hermes Agent mapping for the `models` feature.
//
// Hermes keeps the active provider in a top-level `model:` block (`default`,
// `provider`, `base_url`, `key_env`) and extra OpenAI-compatible endpoints in
// a `custom_providers:` list (`name`, `base_url`, `model`, `models: {<id>:
// {}}`). Per-model entries carry no fields, so limits are dropped on the way
// out. Shape confirmed against a real `~/.hermes/config.yaml`.
import { join } from "node:path";

import {
  HERMESAGENT_CONFIG_FILE_NAME,
  HERMESAGENT_GLOBAL_DIR,
} from "../../constants/hermesagent-paths.js";
import type { SharedWritePath } from "../../lib/shared-file-derive.js";
import { ValidationResult } from "../../types/ai-file.js";
import { readFileContentOrNull } from "../../utils/file.js";
import {
  getHermesagentConfigSharedFileKey,
  getHermesagentRelativeDirPath,
  getHermesagentRulesyncOutputRoot,
  getHermesagentSharedConfigWritePaths,
} from "../../utils/hermesagent.js";
import type { Logger } from "../../utils/logger.js";
import { isPrototypePollutionKey } from "../../utils/prototype-pollution.js";
import { isRecord } from "../../utils/type-guards.js";
import { applySharedConfigPatch, parseSharedConfig } from "../shared/shared-config-gateway.js";
import {
  forTargetModelConfig,
  type ModelsDefault,
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

const HERMESAGENT_GLOBAL_ONLY_MESSAGE =
  "Hermes Agent models is global-only; use --global to sync ~/.hermes/config.yaml";

/** Keys of the `model:` block the models feature owns. */
const HERMES_MODEL_OWNED_KEYS = ["default", "provider", "base_url", "key_env"] as const;

function toHermesCustomProviders(
  providers: Record<string, RulesyncModelProvider>,
  defaultValue: ModelsDefault | undefined,
  logger?: Logger,
): Record<string, unknown>[] {
  const entries: Record<string, unknown>[] = [];
  for (const [id, provider] of Object.entries(providers)) {
    const modelIds = Object.keys(provider.models ?? {});
    if (modelIds.length === 0) {
      logger?.warn(`Hermes models: provider "${id}" names no models, skipping it.`);
      continue;
    }
    const firstModel = modelIds[0] as string;
    const models: Record<string, unknown> = {};
    for (const modelId of modelIds) {
      models[modelId] = {};
    }
    entries.push({
      name: provider.name ?? id,
      ...(provider.api !== undefined && { base_url: provider.api }),
      model: defaultValue?.provider === id ? defaultValue.model : firstModel,
      models,
    });
  }
  return entries;
}

/** Merge generated entries into the existing list by `base_url`. */
function mergeHermesCustomProviders(
  existing: unknown,
  generated: Record<string, unknown>[],
): Record<string, unknown>[] {
  const byBaseUrl = new Map<string, Record<string, unknown>>();
  for (const entry of generated) {
    if (typeof entry.base_url === "string") byBaseUrl.set(entry.base_url, entry);
  }
  const merged: Record<string, unknown>[] = [];
  const replaced = new Set<string>();
  if (Array.isArray(existing)) {
    for (const item of existing) {
      if (!isRecord(item) || typeof item.base_url !== "string") {
        merged.push(item as Record<string, unknown>);
        continue;
      }
      const replacement = byBaseUrl.get(item.base_url);
      if (replacement) {
        replaced.add(item.base_url);
        merged.push(replacement);
      } else {
        merged.push(item);
      }
    }
  }
  for (const entry of generated) {
    if (typeof entry.base_url !== "string" || !replaced.has(entry.base_url)) {
      merged.push(entry);
    }
  }
  return merged;
}

/** Split `<provider>/<model>` on the first `/` only. */
function splitHermesDefault(value: string | undefined): ModelsDefault | undefined {
  if (!value) return undefined;
  const slash = value.indexOf("/");
  if (slash <= 0) return undefined;
  return { provider: value.slice(0, slash), model: value.slice(slash + 1) };
}

function fromHermesCustomProviders(existing: unknown): Record<string, RulesyncModelProvider> {
  const providers: Record<string, RulesyncModelProvider> = {};
  if (!Array.isArray(existing)) return providers;
  for (const item of existing) {
    if (!isRecord(item) || typeof item.name !== "string") continue;
    if (isPrototypePollutionKey(item.name)) continue;
    const models: Record<string, { id: string }> = {};
    if (isRecord(item.models)) {
      for (const modelId of Object.keys(item.models)) {
        if (isPrototypePollutionKey(modelId)) continue;
        models[modelId] = { id: modelId };
      }
    }
    providers[item.name] = {
      name: item.name,
      ...(typeof item.base_url === "string" && { api: item.base_url }),
      models,
    };
  }
  return providers;
}

export class HermesModels extends ToolModels {
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

  override shouldMergeExistingFileContent(): boolean {
    return true;
  }

  override setFileContent(fileContent: string): void {
    const config = parseSharedConfig({ format: "yaml", fileContent });
    const generated = isRecord(this.config.model)
      ? (this.config.model as Record<string, unknown>)
      : {};
    const mergedModel = { ...(isRecord(config.model) ? config.model : {}), ...generated };
    const mergedProviders = mergeHermesCustomProviders(
      config.custom_providers,
      Array.isArray(this.config.custom_providers)
        ? (this.config.custom_providers as Record<string, unknown>[])
        : [],
    );
    const merged = { ...config, model: mergedModel, custom_providers: mergedProviders };

    this.config = merged;
    super.setFileContent(
      applySharedConfigPatch({
        fileKey: getHermesagentConfigSharedFileKey({ global: this.global }),
        feature: "models",
        existingContent: fileContent,
        patch: { model: mergedModel, custom_providers: mergedProviders },
      }),
    );
  }

  override isDeletable(): boolean {
    // config.yaml holds other Hermes settings, so it is never removed.
    return false;
  }

  static getSettablePaths({ global = false }: { global?: boolean } = {}): ToolModelsSettablePaths {
    return {
      relativeDirPath: getHermesagentRelativeDirPath({
        global,
        relativeDirPath: HERMESAGENT_GLOBAL_DIR,
      }),
      relativeFilePath: HERMESAGENT_CONFIG_FILE_NAME,
    };
  }

  /**
   * `config.yaml` under every spelling the global profile root can take.
   * @see getHermesagentSharedConfigWritePaths
   */
  static getExtraSharedWritePaths(): SharedWritePath[] {
    return getHermesagentSharedConfigWritePaths();
  }

  static async fromFile({
    outputRoot = process.cwd(),
    validate = true,
    global = false,
  }: ToolModelsFromFileParams): Promise<HermesModels> {
    if (!global) {
      throw new Error(HERMESAGENT_GLOBAL_ONLY_MESSAGE);
    }
    const paths = this.getSettablePaths({ global });
    const filePath = join(outputRoot, paths.relativeDirPath, paths.relativeFilePath);
    const fileContent = (await readFileContentOrNull(filePath)) ?? "";

    return new HermesModels({
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
  }: ToolModelsFromRulesyncModelsParams): Promise<HermesModels> {
    if (!global) {
      throw new Error(HERMESAGENT_GLOBAL_ONLY_MESSAGE);
    }
    const paths = this.getSettablePaths({ global });
    const fileContent =
      (await readFileContentOrNull(
        join(outputRoot, paths.relativeDirPath, paths.relativeFilePath),
      )) ?? "";
    const config = parseSharedConfig({ format: "yaml", fileContent });

    const { providers, default: defaultValue } = forTargetModelConfig({
      config: rulesyncModels.getJson(),
      toolTarget: "hermesagent",
    });

    // Recompute the owned `model:` keys from the existing block so
    // Hermes-only siblings (e.g. `api_mode`) survive the rewrite.
    const model: Record<string, unknown> = {
      ...(isRecord(config.model) ? config.model : {}),
    };
    for (const key of HERMES_MODEL_OWNED_KEYS) {
      delete model[key];
    }
    if (defaultValue) {
      const active = providers[defaultValue.provider];
      model.default = `${defaultValue.provider}/${defaultValue.model}`;
      model.provider = defaultValue.provider;
      if (typeof active?.api === "string") model.base_url = active.api;
      const keyEnv = active?.env?.[0];
      if (typeof keyEnv === "string") model.key_env = keyEnv;
    }

    const merged = {
      ...config,
      model,
      custom_providers: mergeHermesCustomProviders(
        config.custom_providers,
        toHermesCustomProviders(providers, defaultValue, logger),
      ),
    };

    return new HermesModels({
      outputRoot,
      relativeDirPath: paths.relativeDirPath,
      relativeFilePath: paths.relativeFilePath,
      fileContent: applySharedConfigPatch({
        fileKey: getHermesagentConfigSharedFileKey({ global }),
        feature: "models",
        existingContent: fileContent,
        patch: { model: merged.model, custom_providers: merged.custom_providers },
      }),
      validate,
      global,
    });
  }

  toRulesyncModels(): RulesyncModels {
    const model = isRecord(this.config.model) ? this.config.model : {};
    const providers = fromHermesCustomProviders(this.config.custom_providers);
    // The active provider's connection details live on the `model:` block, not
    // on the list entries: attach them to the matching provider when the entry
    // exists. Anything else (a builtin provider) stays in Hermes.
    const activeName = typeof model.provider === "string" ? model.provider : undefined;
    const active = activeName ? providers[activeName] : undefined;
    if (active) {
      if (typeof model.base_url === "string") active.api = model.base_url;
      if (typeof model.key_env === "string") active.env = [model.key_env];
    }
    const defaultValue =
      typeof model.default === "string" ? splitHermesDefault(model.default) : undefined;
    return this.toRulesyncModelsDefault({
      outputRoot: getHermesagentRulesyncOutputRoot({
        nativeOutputRoot: this.outputRoot,
        global: this.global,
      }),
      fileContent: JSON.stringify(
        {
          providers,
          ...(defaultValue !== undefined && { default: defaultValue }),
        },
        null,
        2,
      ),
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
  }: ToolModelsForDeletionParams): HermesModels {
    return new HermesModels({
      outputRoot,
      relativeDirPath,
      relativeFilePath,
      fileContent: "",
      validate: false,
      global,
    });
  }
}
