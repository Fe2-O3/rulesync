// Qwen Code mapping for the `models` feature.
//
// Providers become `modelProviders` entries (`id`, `name`, `envKey`,
// `baseUrl`, `generationConfig.contextWindowSize`), grouped under the
// provider id. A non-builtin id additionally needs a `providerProtocol`
// mapping to a builtin protocol, or Qwen skips the whole entry — so one is
// written for every managed id that needs it. Credentials are never persisted
// by Qwen; the runtime reads `process.env[envKey]`, hence env names only. The
// default model is `/model` runtime state with no config key.
import { join } from "node:path";

import { z } from "zod/mini";

import { QWENCODE_DIR } from "../../constants/qwencode-paths.js";
import { ValidationResult } from "../../types/ai-file.js";
import { readFileContentOrNull } from "../../utils/file.js";
import { parseJsonc as parseJsoncStrict } from "../../utils/jsonc.js";
import { isPrototypePollutionKey } from "../../utils/prototype-pollution.js";
import { applySharedConfigPatch, sharedConfigFileKey } from "../shared/shared-config-gateway.js";
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

const QWENCODE_SETTINGS_FILE_NAME = "settings.json";

// Builtin ids route to their protocol automatically; anything else needs an
// explicit `providerProtocol` mapping or Qwen skips the entry entirely.
const QWEN_BUILTIN_PROTOCOLS = [
  "openai",
  "openai-responses",
  "anthropic",
  "gemini",
  "vertex-ai",
  "qwen-oauth",
] as const;

const QwenGenerationConfigSchema = z.looseObject({
  contextWindowSize: z.optional(z.number()),
  samplingParams: z.optional(z.looseObject({ max_tokens: z.optional(z.number()) })),
});

const QwenModelSchema = z.looseObject({
  id: z.optional(z.string()),
  name: z.optional(z.string()),
  envKey: z.optional(z.string()),
  baseUrl: z.optional(z.string()),
  generationConfig: z.optional(QwenGenerationConfigSchema),
});

const QwenConfigSchema = z.looseObject({
  $schema: z.optional(z.string()),
  modelProviders: z.optional(z.record(z.string(), z.array(QwenModelSchema))),
  providerProtocol: z.optional(z.record(z.string(), z.string())),
});

type QwenConfig = z.infer<typeof QwenConfigSchema>;
type QwenModelShape = z.infer<typeof QwenModelSchema>;

function toQwenModel(
  model: RulesyncModel,
  provider: RulesyncModelProvider,
): Record<string, unknown> {
  return {
    id: model.id,
    ...(model.name !== undefined && { name: model.name }),
    ...(provider.env?.[0] !== undefined && { envKey: provider.env[0] }),
    ...(provider.api !== undefined && { baseUrl: provider.api }),
    ...((model.limit?.context !== undefined || model.limit?.output !== undefined) && {
      generationConfig: {
        ...(model.limit?.context !== undefined && { contextWindowSize: model.limit.context }),
        ...(model.limit?.output !== undefined && {
          samplingParams: { max_tokens: model.limit.output },
        }),
      },
    }),
  };
}

function fromQwenModel(model: QwenModelShape): {
  id: string;
  model: RulesyncModel;
  envKey?: string;
  baseUrl?: string;
} | null {
  if (!model.id || isPrototypePollutionKey(model.id)) return null;
  const result: RulesyncModel = { id: model.id };
  if (model.name !== undefined) result.name = model.name;
  const context = model.generationConfig?.contextWindowSize;
  const output = model.generationConfig?.samplingParams?.max_tokens;
  if (typeof context === "number" || typeof output === "number") {
    result.limit = {
      ...(typeof context === "number" && { context }),
      ...(typeof output === "number" && { output }),
    };
  }
  return {
    id: model.id,
    model: result,
    ...(model.envKey !== undefined && { envKey: model.envKey }),
    ...(model.baseUrl !== undefined && { baseUrl: model.baseUrl }),
  };
}

export class QwenModels extends ToolModels {
  private readonly json: QwenConfig;

  constructor(params: ToolModelsParams) {
    super(params);
    this.json = QwenConfigSchema.parse(parseJsoncStrict(this.fileContent || "{}"));
  }

  /** settings.json holds all of Qwen's settings, so it is never deleted. */
  override isDeletable(): boolean {
    return false;
  }

  static getSettablePaths(_options: { global?: boolean } = {}): ToolModelsSettablePaths {
    // Same relative path at both scopes; global mode only changes the root.
    return { relativeDirPath: QWENCODE_DIR, relativeFilePath: QWENCODE_SETTINGS_FILE_NAME };
  }

  static async fromFile({
    outputRoot = process.cwd(),
    validate = true,
    global = false,
  }: ToolModelsFromFileParams): Promise<QwenModels> {
    const paths = this.getSettablePaths({ global });
    const fileContent =
      (await readFileContentOrNull(
        join(outputRoot, paths.relativeDirPath, paths.relativeFilePath),
      )) ?? "{}";
    return new QwenModels({
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
  }: ToolModelsFromRulesyncModelsParams): Promise<QwenModels> {
    const paths = this.getSettablePaths({ global });
    const fileContent =
      (await readFileContentOrNull(
        join(outputRoot, paths.relativeDirPath, paths.relativeFilePath),
      )) ?? "";
    const { providers } = forTargetModelConfig({
      config: rulesyncModels.getJson(),
      toolTarget: "qwencode",
    });

    const modelProviders: Record<string, unknown> = {};
    const providerProtocol: Record<string, string> = {};
    for (const [id, provider] of Object.entries(providers)) {
      if (isPrototypePollutionKey(id)) continue;
      const entries = Object.values(provider.models ?? {}).map((model) =>
        toQwenModel(model, provider),
      );
      if (entries.length === 0) {
        logger?.warn(`Qwen models: provider "${id}" names no models, skipping it.`);
        continue;
      }
      modelProviders[id] = entries;
      if (!(QWEN_BUILTIN_PROTOCOLS as readonly string[]).includes(id)) {
        providerProtocol[id] = "openai";
      }
    }

    return new QwenModels({
      outputRoot,
      relativeDirPath: paths.relativeDirPath,
      relativeFilePath: paths.relativeFilePath,
      fileContent: applySharedConfigPatch({
        fileKey: sharedConfigFileKey(paths),
        feature: "models",
        existingContent: fileContent,
        patch: {
          modelProviders,
          ...(Object.keys(providerProtocol).length > 0 && { providerProtocol }),
        },
        filePath: join(outputRoot, paths.relativeDirPath, paths.relativeFilePath),
        logger,
      }),
      validate,
      global,
    });
  }

  toRulesyncModels(): RulesyncModels {
    const providers: Record<string, RulesyncModelProvider> = {};
    for (const [id, entries] of Object.entries(this.json.modelProviders ?? {})) {
      if (isPrototypePollutionKey(id)) continue;
      const models: Record<string, RulesyncModel> = {};
      let api: string | undefined;
      let env: string[] | undefined;
      for (const entry of entries) {
        const converted = fromQwenModel(entry);
        if (!converted) continue;
        models[converted.id] = converted.model;
        if (api === undefined) api = converted.baseUrl;
        if (env === undefined && converted.envKey) env = [converted.envKey];
      }
      providers[id] = {
        ...(api !== undefined && { api }),
        ...(env !== undefined && { env }),
        models,
      };
    }
    return this.toRulesyncModelsDefault({
      fileContent: JSON.stringify({ providers }, null, 2),
    });
  }

  validate(): ValidationResult {
    const json = parseJsoncStrict(this.fileContent || "{}");
    const result = QwenConfigSchema.safeParse(json);
    if (!result.success) {
      return { success: false, error: result.error };
    }
    return { success: true, error: null };
  }

  static forDeletion({
    outputRoot = process.cwd(),
    relativeDirPath,
    relativeFilePath,
    global = false,
  }: ToolModelsForDeletionParams): QwenModels {
    return new QwenModels({
      outputRoot,
      relativeDirPath,
      relativeFilePath,
      fileContent: "{}",
      validate: false,
      global,
    });
  }
}
