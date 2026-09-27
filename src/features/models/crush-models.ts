// Crush mapping for the `models` feature (legacy JSON config).
//
// Crush's primary config is now the Bash `crushrc`, but the JSON file is
// still supported, and rulesync already writes it for MCP. Providers map onto
// the legacy `providers` entries (`base_url`, `api_key` in Crush's `$VAR`
// shell form, `models[]` with `context_window` / `default_max_tokens`), and
// the default lands in the `models.large` slot. Crush requires per-model
// costs, so a model without them is written as 0 with a warning.
import { join } from "node:path";

import { z } from "zod/mini";

import { ValidationResult } from "../../types/ai-file.js";
import { parseJsonc as parseJsoncStrict } from "../../utils/jsonc.js";
import type { Logger } from "../../utils/logger.js";
import { isPrototypePollutionKey } from "../../utils/prototype-pollution.js";
import {
  crushConfigImportContent,
  getCrushConfigSettablePaths,
  parseCrushConfig,
  resolveCrushConfigFile,
} from "../crush-config.js";
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

const CrushModelSchema = z.looseObject({
  id: z.optional(z.string()),
  name: z.optional(z.string()),
  context_window: z.optional(z.number()),
  default_max_tokens: z.optional(z.number()),
  can_reason: z.optional(z.boolean()),
  cost_per_1m_in: z.optional(z.number()),
  cost_per_1m_out: z.optional(z.number()),
});

const CrushProviderSchema = z.looseObject({
  name: z.optional(z.string()),
  base_url: z.optional(z.string()),
  api_key: z.optional(z.string()),
  models: z.optional(z.array(CrushModelSchema)),
});

const CrushSelectedModelSchema = z.looseObject({
  provider: z.optional(z.string()),
  model: z.optional(z.string()),
});

const CrushConfigSchema = z.looseObject({
  $schema: z.optional(z.string()),
  providers: z.optional(z.record(z.string(), CrushProviderSchema)),
  models: z.optional(z.record(z.string(), CrushSelectedModelSchema)),
});

type CrushConfig = z.infer<typeof CrushConfigSchema>;

function toCrushApiKey(env: string[] | undefined): string | undefined {
  const name = env?.[0];
  return name ? `$${name}` : undefined;
}

/** `$VAR` back to `[VAR]`. Anything else is a literal key: never imported. */
function fromCrushApiKey(apiKey: string | undefined): string[] | undefined {
  const match = apiKey ? /^\$([A-Za-z_][A-Za-z0-9_]*)$/.exec(apiKey) : null;
  return match?.[1] ? [match[1]] : undefined;
}

function toCrushModel(model: RulesyncModel, logger: Logger | undefined): Record<string, unknown> {
  if (model.cost?.input === undefined || model.cost?.output === undefined) {
    logger?.warn(
      `Crush models: "${model.id}" names no cost, writing 0 (Crush requires per-model costs).`,
    );
  }
  return {
    id: model.id,
    name: model.name ?? model.id,
    ...(model.limit?.context !== undefined && { context_window: model.limit.context }),
    ...(model.limit?.output !== undefined && { default_max_tokens: model.limit.output }),
    ...(model.tool_call !== undefined && { can_reason: model.tool_call }),
    cost_per_1m_in: model.cost?.input ?? 0,
    cost_per_1m_out: model.cost?.output ?? 0,
  };
}

function toCrushProvider(
  id: string,
  provider: RulesyncModelProvider,
  logger: Logger | undefined,
): Record<string, unknown> {
  const apiKey = toCrushApiKey(provider.env);
  return {
    ...(provider.name !== undefined && { name: provider.name }),
    ...(provider.api !== undefined && { base_url: provider.api }),
    // Custom endpoints speak OpenAI-compatible APIs; Crush's own default
    // (`openai`) applies when no base URL is set.
    ...(provider.api !== undefined && { type: "openai-compat" }),
    ...(apiKey !== undefined && { api_key: apiKey }),
    models: Object.values(provider.models ?? {}).map((model) => toCrushModel(model, logger)),
  };
}

function readExistingModelSlots({
  location,
  outputRoot,
}: {
  location: { fileContent?: string | null; relativeDirPath: string; relativeFilePath: string };
  outputRoot: string;
}): { existingSmall: Record<string, unknown>; hadLarge: boolean } {
  const existingModels = CrushConfigSchema.safeParse(
    parseCrushConfig(
      location.fileContent ?? "{}",
      join(outputRoot, location.relativeDirPath, location.relativeFilePath),
    ),
  );
  if (!existingModels.success) return { existingSmall: {}, hadLarge: false };
  const small = existingModels.data.models?.small;
  return {
    existingSmall:
      typeof small?.provider === "string" && typeof small?.model === "string" ? { small } : {},
    hadLarge: existingModels.data.models?.large !== undefined,
  };
}

type CrushProviderShape = z.infer<typeof CrushProviderSchema>;
type CrushModelShape = z.infer<typeof CrushModelSchema>;

export class CrushModels extends ToolModels {
  private static fromModel(model: CrushModelShape): RulesyncModel | null {
    if (!model.id || isPrototypePollutionKey(model.id)) return null;
    return {
      id: model.id,
      ...(model.name !== undefined && { name: model.name }),
      ...((model.context_window !== undefined || model.default_max_tokens !== undefined) && {
        limit: {
          ...(model.context_window !== undefined && { context: model.context_window }),
          ...(model.default_max_tokens !== undefined && { output: model.default_max_tokens }),
        },
      }),
      ...(model.can_reason !== undefined && { tool_call: model.can_reason }),
      ...((model.cost_per_1m_in !== undefined || model.cost_per_1m_out !== undefined) && {
        cost: {
          ...(model.cost_per_1m_in !== undefined && { input: model.cost_per_1m_in }),
          ...(model.cost_per_1m_out !== undefined && { output: model.cost_per_1m_out }),
        },
      }),
    };
  }

  private static fromProvider(id: string, provider: CrushProviderShape): RulesyncModelProvider {
    const models: Record<string, RulesyncModel> = {};
    for (const model of provider.models ?? []) {
      const converted = CrushModels.fromModel(model);
      if (converted) models[converted.id] = converted;
    }
    const env = fromCrushApiKey(provider.api_key);
    return {
      ...(provider.name !== undefined && { name: provider.name }),
      ...(provider.base_url !== undefined && { api: provider.base_url }),
      ...(env !== undefined && { env }),
      models,
    };
  }

  private readonly json: CrushConfig;

  constructor(params: ToolModelsParams) {
    super(params);
    this.json = CrushConfigSchema.parse(
      parseCrushConfig(this.fileContent || "{}", join(this.relativeDirPath, this.relativeFilePath)),
    );
  }

  /** crush.json is Crush's primary config file, so it is never deleted. */
  override isDeletable(): boolean {
    return false;
  }

  static getSettablePaths({ global = false }: { global?: boolean } = {}): ToolModelsSettablePaths {
    return getCrushConfigSettablePaths({ global });
  }

  static async fromFile({
    outputRoot = process.cwd(),
    validate = true,
    global = false,
  }: ToolModelsFromFileParams): Promise<CrushModels> {
    const location = await resolveCrushConfigFile({ outputRoot, global });
    return new CrushModels({
      outputRoot,
      relativeDirPath: location.relativeDirPath,
      relativeFilePath: location.relativeFilePath,
      fileContent: crushConfigImportContent(location),
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
  }: ToolModelsFromRulesyncModelsParams): Promise<CrushModels> {
    const location = await resolveCrushConfigFile({ outputRoot, global });
    const { providers, default: defaultValue } = forTargetModelConfig({
      config: rulesyncModels.getJson(),
      toolTarget: "crush",
    });

    const crushProviders: Record<string, unknown> = {};
    for (const [id, provider] of Object.entries(providers)) {
      if (isPrototypePollutionKey(id)) continue;
      crushProviders[id] = toCrushProvider(id, provider, logger);
    }
    // The `small` slot is the user's; carry it over so replacing the owned
    // `models` key neither deletes it nor leaves a retracted `large` behind.
    const { existingSmall, hadLarge } = readExistingModelSlots({ location, outputRoot });
    const large = defaultValue
      ? { large: { provider: defaultValue.provider, model: defaultValue.model } }
      : {};

    return new CrushModels({
      outputRoot,
      relativeDirPath: location.relativeDirPath,
      relativeFilePath: location.relativeFilePath,
      fileContent: applySharedConfigPatch({
        fileKey: sharedConfigFileKey(this.getSettablePaths({ global })),
        feature: "models",
        existingContent: location.fileContent ?? "",
        patch: {
          providers: crushProviders,
          ...((Object.keys(existingSmall).length > 0 ||
            Object.keys(large).length > 0 ||
            hadLarge) && {
            models: { ...existingSmall, ...large },
          }),
        },
        filePath: join(outputRoot, location.relativeDirPath, location.relativeFilePath),
        logger,
      }),
      validate,
      global,
    });
  }

  toRulesyncModels(): RulesyncModels {
    const providers: Record<string, RulesyncModelProvider> = {};
    for (const [id, provider] of Object.entries(this.json.providers ?? {})) {
      if (isPrototypePollutionKey(id)) continue;
      providers[id] = CrushModels.fromProvider(id, provider);
    }
    const large = this.json.models?.large;
    const defaultValue =
      large?.provider && large?.model
        ? { provider: large.provider, model: large.model }
        : undefined;
    return this.toRulesyncModelsDefault({
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
    const json = parseJsoncStrict(this.fileContent || "{}");
    const result = CrushConfigSchema.safeParse(json);
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
  }: ToolModelsForDeletionParams): CrushModels {
    return new CrushModels({
      outputRoot,
      relativeDirPath,
      relativeFilePath,
      fileContent: "{}",
      validate: false,
      global,
    });
  }
}
