// Pi Coding Agent mapping for the `models` feature (global-only).
//
// Providers become entries of the `providers` map in the dedicated
// `~/.pi/agent/models.json` (`baseUrl`, `api: "openai-completions"`, `apiKey`
// in pi's `$VAR` form, `models[]` with `contextWindow` / `maxTokens`). Keys
// stay out unless an env var names them — pi loads keyless entries and
// resolves auth via `/login`, `auth.json`, or CLI flags. pi's own defaults
// apply where the source is silent (128000 context, 16384 max tokens, zero
// costs); only costs are written out, matching pi's own examples. The default
// model (`defaultModel` in `settings.json`) is a different file and stays
// untouched.
import { join } from "node:path";

import { z } from "zod/mini";

import { PI_AGENT_DIR_PATH } from "../../constants/pi-paths.js";
import { ValidationResult } from "../../types/ai-file.js";
import { readFileContentOrNull } from "../../utils/file.js";
import { parseJsonc as parseJsoncStrict } from "../../utils/jsonc.js";
import { isPrototypePollutionKey } from "../../utils/prototype-pollution.js";
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

const PI_MODELS_FILE_NAME = "models.json";

const PiCostSchema = z.looseObject({
  input: z.optional(z.number()),
  output: z.optional(z.number()),
});

const PiModelSchema = z.looseObject({
  id: z.optional(z.string()),
  name: z.optional(z.string()),
  reasoning: z.optional(z.boolean()),
  input: z.optional(z.array(z.string())),
  contextWindow: z.optional(z.number()),
  maxTokens: z.optional(z.number()),
  cost: z.optional(PiCostSchema),
});

const PiProviderSchema = z.looseObject({
  baseUrl: z.optional(z.string()),
  api: z.optional(z.string()),
  apiKey: z.optional(z.string()),
  models: z.optional(z.array(PiModelSchema)),
});

const PiModelsFileSchema = z.looseObject({
  $schema: z.optional(z.string()),
  providers: z.optional(z.record(z.string(), PiProviderSchema)),
});

type PiModelsFile = z.infer<typeof PiModelsFileSchema>;
type PiModelShape = z.infer<typeof PiModelSchema>;

function toPiApiKey(env: string[] | undefined): string | undefined {
  const name = env?.[0];
  return name ? `$${name}` : undefined;
}

function fromPiApiKey(apiKey: string | undefined): string[] | undefined {
  const match = apiKey ? /^\$([A-Za-z_][A-Za-z0-9_]*)$/.exec(apiKey) : null;
  return match?.[1] ? [match[1]] : undefined;
}

function toPiModel(model: RulesyncModel): Record<string, unknown> {
  const images = model.modalities?.input?.includes("image") === true;
  return {
    id: model.id,
    ...(model.name !== undefined && { name: model.name }),
    reasoning: model.reasoning ?? false,
    input: images ? ["text", "image"] : ["text"],
    ...(model.limit?.context !== undefined && { contextWindow: model.limit.context }),
    ...(model.limit?.output !== undefined && { maxTokens: model.limit.output }),
    cost: {
      input: model.cost?.input ?? 0,
      output: model.cost?.output ?? 0,
    },
  };
}

function fromPiModel(model: PiModelShape): RulesyncModel | null {
  if (!model.id || isPrototypePollutionKey(model.id)) return null;
  const result: RulesyncModel = { id: model.id };
  if (model.name !== undefined) result.name = model.name;
  if (typeof model.reasoning === "boolean") result.reasoning = model.reasoning;
  if (Array.isArray(model.input) && model.input.includes("image")) {
    result.modalities = { input: ["image"] };
  }
  if (typeof model.contextWindow === "number" || typeof model.maxTokens === "number") {
    result.limit = {
      ...(typeof model.contextWindow === "number" && { context: model.contextWindow }),
      ...(typeof model.maxTokens === "number" && { output: model.maxTokens }),
    };
  }
  const cost = model.cost;
  if (cost && (typeof cost.input === "number" || typeof cost.output === "number")) {
    result.cost = {
      ...(typeof cost.input === "number" && { input: cost.input }),
      ...(typeof cost.output === "number" && { output: cost.output }),
    };
  }
  return result;
}

export class PiModels extends ToolModels {
  private readonly json: PiModelsFile;

  constructor(params: ToolModelsParams) {
    super(params);
    this.json = PiModelsFileSchema.parse(JSON.parse(this.fileContent || "{}"));
  }

  static getSettablePaths({ global = false }: { global?: boolean } = {}): ToolModelsSettablePaths {
    if (!global) {
      throw new Error("Pi models is global-only; use --global to sync ~/.pi/agent/models.json");
    }
    return { relativeDirPath: PI_AGENT_DIR_PATH, relativeFilePath: PI_MODELS_FILE_NAME };
  }

  static async fromFile({
    outputRoot = process.cwd(),
    validate = true,
    global = false,
  }: ToolModelsFromFileParams): Promise<PiModels> {
    const paths = this.getSettablePaths({ global });
    const fileContent =
      (await readFileContentOrNull(
        join(outputRoot, paths.relativeDirPath, paths.relativeFilePath),
      )) ?? "{}";
    return new PiModels({
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
  }: ToolModelsFromRulesyncModelsParams): Promise<PiModels> {
    const paths = this.getSettablePaths({ global });
    const { providers } = forTargetModelConfig({
      config: rulesyncModels.getJson(),
      toolTarget: "pi",
    });

    const piProviders: Record<string, unknown> = {};
    for (const [id, provider] of Object.entries(providers)) {
      if (isPrototypePollutionKey(id)) continue;
      const models = Object.values(provider.models ?? {}).map((model) => toPiModel(model));
      if (models.length === 0) {
        logger?.warn(`Pi models: provider "${id}" names no models, skipping it.`);
        continue;
      }
      const apiKey = toPiApiKey(provider.env);
      piProviders[id] = {
        ...(provider.name !== undefined && { name: provider.name }),
        ...(provider.api !== undefined && { baseUrl: provider.api }),
        api: "openai-completions",
        ...(apiKey !== undefined && { apiKey }),
        models,
      };
    }

    return new PiModels({
      outputRoot,
      relativeDirPath: paths.relativeDirPath,
      relativeFilePath: paths.relativeFilePath,
      fileContent: JSON.stringify({ providers: piProviders }, null, 2),
      validate,
      global,
    });
  }

  toRulesyncModels(): RulesyncModels {
    const providers: Record<string, RulesyncModelProvider> = {};
    for (const [id, provider] of Object.entries(this.json.providers ?? {})) {
      if (isPrototypePollutionKey(id)) continue;
      const models: Record<string, RulesyncModel> = {};
      for (const model of provider.models ?? []) {
        const converted = fromPiModel(model);
        if (converted) models[converted.id] = converted;
      }
      const env = fromPiApiKey(provider.apiKey);
      providers[id] = {
        ...(provider.baseUrl !== undefined && { api: provider.baseUrl }),
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
    const result = PiModelsFileSchema.safeParse(json);
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
  }: ToolModelsForDeletionParams): PiModels {
    return new PiModels({
      outputRoot,
      relativeDirPath,
      relativeFilePath,
      fileContent: "{}",
      validate: false,
      global,
    });
  }
}
