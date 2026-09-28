// Aider mapping for the `models` feature.
//
// Each model becomes an entry of the flat `.aider.model.metadata.json` map
// (`max_input_tokens` / `max_output_tokens`, per-token costs, `mode: chat`).
// Aider routes OpenAI-compatible endpoints through the `openai/` model prefix
// (`aider --model openai/<id>` with `OPENAI_API_BASE` set), so providers with
// an endpoint use that prefix and `litellm_provider: "openai"`; providers
// without one keep the `<provider>/<model>` key. Costs convert from per-1M
// (models.dev) to per-token. A model without a context window is skipped.
// The default model is a CLI flag with no file key.
import { join } from "node:path";

import { z } from "zod/mini";

import { ValidationResult } from "../../types/ai-file.js";
import { readFileContentOrNull } from "../../utils/file.js";
import { parseJsonc as parseJsoncStrict } from "../../utils/jsonc.js";
import type { Logger } from "../../utils/logger.js";
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

const AIDER_METADATA_FILE_NAME = ".aider.model.metadata.json";

const AiderModelSchema = z.looseObject({
  max_tokens: z.optional(z.number()),
  max_input_tokens: z.optional(z.number()),
  max_output_tokens: z.optional(z.number()),
  input_cost_per_token: z.optional(z.number()),
  output_cost_per_token: z.optional(z.number()),
  litellm_provider: z.optional(z.string()),
  mode: z.optional(z.string()),
});

const AiderMetadataSchema = z.record(z.string(), AiderModelSchema);

function toAiderKey(providerId: string, modelId: string, hasEndpoint: boolean): string {
  return hasEndpoint ? `openai/${modelId}` : `${providerId}/${modelId}`;
}

function toAiderModel(
  model: RulesyncModel,
  logger: Logger | undefined,
): Record<string, unknown> | null {
  if (model.limit?.context === undefined) {
    logger?.warn(`Aider models: "${model.id}" names no context window, skipping it.`);
    return null;
  }
  return {
    ...(model.limit?.output !== undefined && {
      max_tokens: model.limit.output,
      max_output_tokens: model.limit.output,
    }),
    max_input_tokens: model.limit.context,
    ...(model.cost?.input !== undefined && { input_cost_per_token: model.cost.input / 1_000_000 }),
    ...(model.cost?.output !== undefined && {
      output_cost_per_token: model.cost.output / 1_000_000,
    }),
    mode: "chat",
  };
}

function fromAiderModel(id: string, model: z.infer<typeof AiderModelSchema>): RulesyncModel {
  const result: RulesyncModel = { id };
  if (typeof model.max_input_tokens === "number" || typeof model.max_output_tokens === "number") {
    result.limit = {
      ...(typeof model.max_input_tokens === "number" && { context: model.max_input_tokens }),
      ...(typeof model.max_output_tokens === "number" && { output: model.max_output_tokens }),
    };
  }
  if (
    typeof model.input_cost_per_token === "number" ||
    typeof model.output_cost_per_token === "number"
  ) {
    result.cost = {
      ...(typeof model.input_cost_per_token === "number" && {
        input: model.input_cost_per_token * 1_000_000,
      }),
      ...(typeof model.output_cost_per_token === "number" && {
        output: model.output_cost_per_token * 1_000_000,
      }),
    };
  }
  return result;
}

export class AiderModels extends ToolModels {
  private readonly json: Record<string, z.infer<typeof AiderModelSchema>>;

  constructor(params: ToolModelsParams) {
    super(params);
    this.json = AiderMetadataSchema.parse(JSON.parse(this.fileContent || "{}"));
  }

  static getSettablePaths(_options: { global?: boolean } = {}): ToolModelsSettablePaths {
    // Aider reads the home file, the git root, and the launch directory; the
    // project file and the home file are the two rulesync scopes.
    return { relativeDirPath: ".", relativeFilePath: AIDER_METADATA_FILE_NAME };
  }

  static async fromFile({
    outputRoot = process.cwd(),
    validate = true,
    global = false,
  }: ToolModelsFromFileParams): Promise<AiderModels> {
    const paths = this.getSettablePaths({ global });
    const fileContent =
      (await readFileContentOrNull(
        join(outputRoot, paths.relativeDirPath, paths.relativeFilePath),
      )) ?? "{}";
    return new AiderModels({
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
  }: ToolModelsFromRulesyncModelsParams): Promise<AiderModels> {
    const paths = this.getSettablePaths({ global });
    const { providers } = forTargetModelConfig({
      config: rulesyncModels.getJson(),
      toolTarget: "aider",
    });

    const entries: Record<string, unknown> = {};
    const seen = new Set<string>();
    for (const [id, provider] of Object.entries(providers)) {
      if (isPrototypePollutionKey(id)) continue;
      const hasEndpoint = provider.api !== undefined;
      for (const model of Object.values(provider.models ?? {})) {
        if (isPrototypePollutionKey(model.id)) continue;
        const converted = toAiderModel(model, logger);
        if (!converted) continue;
        const key = toAiderKey(id, model.id, hasEndpoint);
        if (seen.has(key)) {
          logger?.warn(`Aider models: duplicate entry "${key}", keeping the first.`);
          continue;
        }
        seen.add(key);
        entries[key] = {
          ...converted,
          litellm_provider: hasEndpoint ? "openai" : id,
        };
      }
    }

    return new AiderModels({
      outputRoot,
      relativeDirPath: paths.relativeDirPath,
      relativeFilePath: paths.relativeFilePath,
      fileContent: JSON.stringify(entries, null, 2),
      validate,
      global,
    });
  }

  toRulesyncModels(): RulesyncModels {
    const providers: Record<string, RulesyncModelProvider> = {};
    for (const [key, model] of Object.entries(this.json)) {
      if (isPrototypePollutionKey(key)) continue;
      // Keys are `<provider>/<model>`; the endpoint binding is not
      // recoverable from this file (it lives in env/CLI flags), so imported
      // providers carry limits and costs only.
      const slash = key.indexOf("/");
      if (slash <= 0) continue;
      const providerId = key.slice(0, slash);
      const modelId = key.slice(slash + 1);
      if (isPrototypePollutionKey(providerId) || isPrototypePollutionKey(modelId)) continue;
      const converted = fromAiderModel(modelId, model);
      const existing: RulesyncModelProvider = providers[providerId] ?? { models: {} };
      existing.models = { ...existing.models, [modelId]: converted };
      providers[providerId] = existing;
    }
    return this.toRulesyncModelsDefault({
      fileContent: JSON.stringify({ providers }, null, 2),
    });
  }

  validate(): ValidationResult {
    const json = parseJsoncStrict(this.fileContent || "{}");
    const result = AiderMetadataSchema.safeParse(json);
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
  }: ToolModelsForDeletionParams): AiderModels {
    return new AiderModels({
      outputRoot,
      relativeDirPath,
      relativeFilePath,
      fileContent: "{}",
      validate: false,
      global,
    });
  }
}
