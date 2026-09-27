import { readdir } from "node:fs/promises";
// Goose mapping for the `models` feature (global-only).
//
// Each provider becomes one `custom_providers/<id>.json` file (`engine:
// openai`, `api_key_env`, `base_url`, `models[]` with `context_limit`), and
// the default lands in `config.yaml` as `active_provider` plus a
// `providers.<id>` entry merged over the existing map. Filenames and the
// `name` field both carry the provider id, so import splits nothing and
// round-trips exactly. Stale provider files are never auto-deleted: the
// directory holds hand-made providers too.
import { join } from "node:path";

import { z } from "zod/mini";

import { GOOSE_GLOBAL_DIR } from "../../constants/goose-paths.js";
import { ValidationResult } from "../../types/ai-file.js";
import { ToolFile } from "../../types/tool-file.js";
import { readFileContentOrNull } from "../../utils/file.js";
import type { Logger } from "../../utils/logger.js";
import { isPrototypePollutionKey } from "../../utils/prototype-pollution.js";
import { isRecord } from "../../utils/type-guards.js";
import { applySharedConfigPatch, parseSharedConfig } from "../shared/shared-config-gateway.js";
import {
  forTargetModelConfig,
  type ModelsDefault,
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

const GOOSE_GLOBAL_ONLY_MESSAGE =
  "Goose models is global-only; use --global to sync ~/.config/goose/";

const GOOSE_CUSTOM_PROVIDERS_DIR = join(GOOSE_GLOBAL_DIR, "custom_providers");
const GOOSE_CONFIG_FILE_NAME = "config.yaml";

const GooseModelSchema = z.looseObject({
  name: z.optional(z.string()),
  context_limit: z.optional(z.number()),
  reasoning: z.optional(z.boolean()),
  // Goose's own writer emits explicit nulls; they mean absent.
  input_token_cost: z.optional(z.nullable(z.number())),
  output_token_cost: z.optional(z.nullable(z.number())),
});

const GooseProviderSchema = z.looseObject({
  name: z.optional(z.string()),
  engine: z.optional(z.string()),
  display_name: z.optional(z.string()),
  api_key_env: z.optional(z.string()),
  base_url: z.optional(z.string()),
  models: z.optional(z.array(GooseModelSchema)),
});

type GooseProviderFileContent = z.infer<typeof GooseProviderSchema>;

function toGooseModel(
  model: RulesyncModel,
  logger: Logger | undefined,
): Record<string, unknown> | null {
  if (model.limit?.context === undefined) {
    logger?.warn(`Goose models: "${model.id}" names no context window, skipping it.`);
    return null;
  }
  return {
    name: model.id,
    context_limit: model.limit.context,
    input_token_cost: model.cost?.input ?? null,
    output_token_cost: model.cost?.output ?? null,
    reasoning: model.reasoning ?? false,
  };
}

function toGooseProviderFile(
  id: string,
  provider: RulesyncModelProvider,
  logger: Logger | undefined,
): { fileName: string; content: Record<string, unknown> } | null {
  const models: Record<string, unknown>[] = [];
  for (const model of Object.values(provider.models ?? {})) {
    const converted = toGooseModel(model, logger);
    if (converted) models.push(converted);
  }
  if (models.length === 0) {
    logger?.warn(`Goose models: provider "${id}" has no writable models, skipping it.`);
    return null;
  }
  return {
    fileName: `${id}.json`,
    content: {
      name: id,
      engine: "openai",
      display_name: provider.name ?? id,
      api_key_env: provider.env?.[0] ?? "",
      ...(provider.api !== undefined && { base_url: provider.api }),
      models,
    },
  };
}

function isCostNumber(value: unknown): value is number {
  return typeof value === "number";
}

function fromGooseProviderFile(json: GooseProviderFileContent): {
  id: string;
  provider: RulesyncModelProvider;
} | null {
  const id = typeof json.name === "string" ? json.name : undefined;
  if (!id || isPrototypePollutionKey(id)) return null;
  const models: Record<string, RulesyncModel> = {};
  for (const model of json.models ?? []) {
    const modelId = model.name;
    if (!modelId || isPrototypePollutionKey(modelId)) continue;
    models[modelId] = {
      id: modelId,
      ...(typeof model.context_limit === "number" && { limit: { context: model.context_limit } }),
      ...(model.reasoning !== undefined && { reasoning: model.reasoning }),
      ...((isCostNumber(model.input_token_cost) || isCostNumber(model.output_token_cost)) && {
        cost: {
          ...(isCostNumber(model.input_token_cost) && { input: model.input_token_cost }),
          ...(isCostNumber(model.output_token_cost) && { output: model.output_token_cost }),
        },
      }),
    };
  }
  return {
    id,
    provider: {
      ...(typeof json.display_name === "string" && { name: json.display_name }),
      ...(typeof json.base_url === "string" && { api: json.base_url }),
      ...(typeof json.api_key_env === "string" &&
        json.api_key_env !== "" && { env: [json.api_key_env] }),
      models,
    },
  };
}

/** The per-provider JSON files beside `config.yaml`. Dedicated files, written whole. */
export class GooseProviderFile extends ToolFile {
  constructor(
    params: ToolModelsParams & {
      validate?: boolean;
    },
  ) {
    super({ ...params, validate: false });
    if (params.validate) {
      const result = this.validate();
      if (!result.success) {
        throw result.error;
      }
    }
  }

  override isDeletable(): boolean {
    // The directory holds hand-made providers too; a removed source entry
    // stops updating its file rather than deleting a file the user may own.
    return false;
  }

  validate(): ValidationResult {
    const json = JSON.parse(this.fileContent || "{}");
    const result = GooseProviderSchema.safeParse(json);
    if (!result.success) {
      return { success: false, error: result.error };
    }
    return { success: true, error: null };
  }

  toRulesyncModels(): RulesyncModels {
    const converted = fromGooseProviderFile(
      GooseProviderSchema.parse(JSON.parse(this.fileContent || "{}")),
    );
    const providers: Record<string, RulesyncModelProvider> = {};
    if (converted) providers[converted.id] = converted.provider;
    return new RulesyncModels({
      outputRoot: this.outputRoot,
      relativeDirPath: ".rulesync",
      relativeFilePath: "models.jsonc",
      fileContent: JSON.stringify({ providers }, null, 2),
      validate: false,
    });
  }
}

/** The shared `config.yaml`: carries `active_provider` and the providers map. */
export class GooseModels extends ToolModels {
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

  /** config.yaml is Goose's primary user file, so it is never deleted. */
  override isDeletable(): boolean {
    return false;
  }

  static getSettablePaths(_options: { global?: boolean } = {}): ToolModelsSettablePaths {
    return { relativeDirPath: GOOSE_GLOBAL_DIR, relativeFilePath: GOOSE_CONFIG_FILE_NAME };
  }

  static async fromFile({
    outputRoot = process.cwd(),
    validate = true,
    global = false,
  }: ToolModelsFromFileParams): Promise<GooseModels> {
    if (!global) {
      throw new Error(GOOSE_GLOBAL_ONLY_MESSAGE);
    }
    const paths = this.getSettablePaths({ global });
    const fileContent =
      (await readFileContentOrNull(
        join(outputRoot, paths.relativeDirPath, paths.relativeFilePath),
      )) ?? "";
    return new GooseModels({
      outputRoot,
      relativeDirPath: paths.relativeDirPath,
      relativeFilePath: paths.relativeFilePath,
      fileContent,
      validate,
      global,
    });
  }

  /** Every provider file plus the shared config, for the import direction. */
  static async loadAllFiles({
    outputRoot = process.cwd(),
    global = false,
  }: {
    outputRoot?: string;
    global?: boolean;
  }): Promise<ToolFile[]> {
    if (!global) {
      throw new Error(GOOSE_GLOBAL_ONLY_MESSAGE);
    }
    const files: ToolFile[] = [
      await GooseModels.fromFile({ outputRoot, global, logger: undefined }),
    ];
    let entries: string[];
    try {
      entries = await readdir(join(outputRoot, GOOSE_CUSTOM_PROVIDERS_DIR));
    } catch {
      return files;
    }
    for (const entry of entries.toSorted()) {
      if (!entry.endsWith(".json") || entry.startsWith(".")) continue;
      const fileContent = await readFileContentOrNull(
        join(outputRoot, GOOSE_CUSTOM_PROVIDERS_DIR, entry),
      );
      if (fileContent === null) continue;
      files.push(
        new GooseProviderFile({
          outputRoot,
          relativeDirPath: GOOSE_CUSTOM_PROVIDERS_DIR,
          relativeFilePath: entry,
          fileContent,
          validate: true,
        }),
      );
    }
    return files;
  }

  /**
   * One provider file per source provider, for the generate direction.
   * Mirrors the hooks processor's auxiliary-file hook.
   */
  static async getAuxiliaryFiles({
    rulesyncModels,
    outputRoot = process.cwd(),
    global = false,
    logger,
  }: {
    rulesyncModels: RulesyncModels;
    outputRoot?: string;
    global?: boolean;
    logger?: Logger;
  }): Promise<ToolFile[]> {
    if (!global) {
      throw new Error(GOOSE_GLOBAL_ONLY_MESSAGE);
    }
    const { providers } = forTargetModelConfig({
      config: rulesyncModels.getJson(),
      toolTarget: "goose",
    });
    const files: ToolFile[] = [];
    for (const [id, provider] of Object.entries(providers)) {
      if (isPrototypePollutionKey(id)) continue;
      const converted = toGooseProviderFile(id, provider, logger);
      if (!converted) continue;
      files.push(
        new GooseProviderFile({
          outputRoot,
          relativeDirPath: GOOSE_CUSTOM_PROVIDERS_DIR,
          relativeFilePath: converted.fileName,
          fileContent: JSON.stringify(converted.content, null, 2),
          validate: true,
        }),
      );
    }
    return files;
  }

  static async fromRulesyncModels({
    outputRoot = process.cwd(),
    rulesyncModels,
    validate = true,
    global = false,
    logger,
  }: ToolModelsFromRulesyncModelsParams): Promise<GooseModels> {
    if (!global) {
      throw new Error(GOOSE_GLOBAL_ONLY_MESSAGE);
    }
    const paths = this.getSettablePaths({ global });
    const fileContent =
      (await readFileContentOrNull(
        join(outputRoot, paths.relativeDirPath, paths.relativeFilePath),
      )) ?? "";

    const { providers, default: defaultValue } = forTargetModelConfig({
      config: rulesyncModels.getJson(),
      toolTarget: "goose",
    });
    const config = parseSharedConfig({ format: "yaml", fileContent });
    const existingProviders = isRecord(config.providers) ? config.providers : {};
    const mergedProviders: Record<string, unknown> = { ...existingProviders };
    for (const [id] of Object.entries(providers)) {
      if (isPrototypePollutionKey(id)) continue;
      mergedProviders[id] = {
        ...(isRecord(existingProviders[id]) ? existingProviders[id] : {}),
        enabled: true,
        configured: true,
        ...(defaultValue?.provider === id && { model: defaultValue.model }),
      };
    }

    return new GooseModels({
      outputRoot,
      relativeDirPath: paths.relativeDirPath,
      relativeFilePath: paths.relativeFilePath,
      fileContent: applySharedConfigPatch({
        fileKey: join(GOOSE_GLOBAL_DIR, GOOSE_CONFIG_FILE_NAME),
        feature: "models",
        existingContent: fileContent,
        patch: {
          ...(defaultValue !== undefined && { active_provider: defaultValue.provider }),
          providers: mergedProviders,
        },
        filePath: join(outputRoot, paths.relativeDirPath, paths.relativeFilePath),
        logger,
      }),
      validate,
      global,
    });
  }

  toRulesyncModels(): RulesyncModels {
    const defaultValue = GooseModels.defaultFromConfig(this.config);
    return this.toRulesyncModelsDefault({
      fileContent: JSON.stringify(
        {
          providers: {},
          ...(defaultValue !== undefined && { default: defaultValue }),
        },
        null,
        2,
      ),
    });
  }

  private static defaultFromConfig(config: Record<string, unknown>): ModelsDefault | undefined {
    const active = typeof config.active_provider === "string" ? config.active_provider : undefined;
    if (!active || isPrototypePollutionKey(active)) return undefined;
    const entry = isRecord(config.providers) ? config.providers[active] : undefined;
    const model = isRecord(entry) && typeof entry.model === "string" ? entry.model : undefined;
    if (!model) return undefined;
    return { provider: active, model };
  }

  validate(): ValidationResult {
    return { success: true, error: null };
  }

  static forDeletion({
    outputRoot = process.cwd(),
    relativeDirPath,
    relativeFilePath,
    global = false,
  }: ToolModelsForDeletionParams): GooseModels {
    return new GooseModels({
      outputRoot,
      relativeDirPath,
      relativeFilePath,
      fileContent: "",
      validate: false,
      global,
    });
  }
}
