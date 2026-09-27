// Zed mapping for the `models` feature.
//
// Providers become `language_models.openai_compatible` entries (`api_url` plus
// `available_models[]` with `display_name`, the required `max_tokens` context
// window, and `capabilities`). Two deliberate omissions, both documented in
// Zed's own docs: API keys are never written (Zed reads them from the keychain
// or a generated `<ID>_API_KEY` variable), and the default model stays in
// Zed's UI (its `agent` key is owned by the permissions feature). A model
// without a context window is skipped: `max_tokens` is required and never
// guessed.
import { join } from "node:path";

import { z } from "zod/mini";

import {
  ZED_DIR,
  ZED_SETTINGS_FILE_NAME,
  getZedGlobalDir,
  getZedOtherPlatformGlobalDir,
} from "../../constants/zed-paths.js";
import type { SharedWritePath } from "../../lib/shared-file-derive.js";
import { ValidationResult } from "../../types/ai-file.js";
import { readFileContentOrNull } from "../../utils/file.js";
import { parseJsonc as parseJsoncStrict } from "../../utils/jsonc.js";
import { isPrototypePollutionKey } from "../../utils/prototype-pollution.js";
import type { Logger } from "../../utils/logger.js";
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

const ZedModelSchema = z.looseObject({
  name: z.optional(z.string()),
  display_name: z.optional(z.string()),
  max_tokens: z.optional(z.number()),
  max_output_tokens: z.optional(z.number()),
  capabilities: z.optional(
    z.looseObject({
      tools: z.optional(z.boolean()),
      images: z.optional(z.boolean()),
    }),
  ),
});

const ZedProviderSchema = z.looseObject({
  api_url: z.optional(z.string()),
  available_models: z.optional(z.array(ZedModelSchema)),
});

const ZedConfigSchema = z.looseObject({
  $schema: z.optional(z.string()),
  language_models: z.optional(
    z.looseObject({
      openai_compatible: z.optional(z.record(z.string(), ZedProviderSchema)),
    }),
  ),
});

type ZedConfig = z.infer<typeof ZedConfigSchema>;
type ZedModelShape = z.infer<typeof ZedModelSchema>;

function toZedModel(
  model: RulesyncModel,
  logger: Logger | undefined,
): Record<string, unknown> | null {
  if (model.limit?.context === undefined) {
    logger?.warn(
      `Zed models: "${model.id}" names no context window, skipping it (max_tokens is required).`,
    );
    return null;
  }
  const images = model.modalities?.input?.includes("image");
  return {
    name: model.id,
    ...(model.name !== undefined && { display_name: model.name }),
    max_tokens: model.limit.context,
    ...(model.limit?.output !== undefined && { max_output_tokens: model.limit.output }),
    ...((model.tool_call === false || images === true) && {
      capabilities: {
        ...(model.tool_call === false && { tools: false }),
        ...(images === true && { images: true }),
      },
    }),
  };
}

function fromZedModel(model: ZedModelShape, fallbackId: string): RulesyncModel | null {
  const id = model.name ?? fallbackId;
  if (!id || isPrototypePollutionKey(id)) return null;
  return {
    id,
    ...(model.display_name !== undefined && { name: model.display_name }),
    ...(model.max_tokens !== undefined && {
      limit: {
        context: model.max_tokens,
        ...(model.max_output_tokens !== undefined && { output: model.max_output_tokens }),
      },
    }),
    ...(model.capabilities?.tools !== undefined && { tool_call: model.capabilities.tools }),
    ...(model.capabilities?.images === true && { modalities: { input: ["image"] } }),
  };
}

export class ZedModels extends ToolModels {
  private readonly json: ZedConfig;

  constructor(params: ToolModelsParams) {
    super(params);
    this.json = ZedConfigSchema.parse(JSON.parse(this.fileContent || "{}"));
  }

  /** settings.json holds all of Zed's settings, so it is never deleted. */
  override isDeletable(): boolean {
    return false;
  }

  static getSettablePaths({ global = false }: { global?: boolean } = {}): ToolModelsSettablePaths {
    return {
      relativeDirPath: global ? getZedGlobalDir() : ZED_DIR,
      relativeFilePath: ZED_SETTINGS_FILE_NAME,
    };
  }

  /** @see getZedOtherPlatformGlobalDir */
  static getExtraSharedWritePaths({
    global = false,
  }: { global?: boolean } = {}): SharedWritePath[] {
    if (!global) {
      return [];
    }
    return [
      {
        relativeDirPath: getZedOtherPlatformGlobalDir(),
        relativeFilePath: ZED_SETTINGS_FILE_NAME,
      },
    ];
  }

  static async fromFile({
    outputRoot = process.cwd(),
    validate = true,
    global = false,
  }: ToolModelsFromFileParams): Promise<ZedModels> {
    const paths = this.getSettablePaths({ global });
    const fileContent =
      (await readFileContentOrNull(join(outputRoot, paths.relativeDirPath, paths.relativeFilePath))) ??
      "{}";
    return new ZedModels({
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
  }: ToolModelsFromRulesyncModelsParams): Promise<ZedModels> {
    const paths = this.getSettablePaths({ global });
    const fileContent =
      (await readFileContentOrNull(join(outputRoot, paths.relativeDirPath, paths.relativeFilePath))) ??
      "";
    const { providers } = forTargetModelConfig({
      config: rulesyncModels.getJson(),
      toolTarget: "zed",
    });

    const openaiCompatible: Record<string, unknown> = {};
    for (const [id, provider] of Object.entries(providers)) {
      if (isPrototypePollutionKey(id)) continue;
      const availableModels: Record<string, unknown>[] = [];
      for (const model of Object.values(provider.models ?? {})) {
        const converted = toZedModel(model, logger);
        if (converted) availableModels.push(converted);
      }
      if (availableModels.length === 0) {
        logger?.warn(`Zed models: provider "${id}" has no writable models, skipping it.`);
        continue;
      }
      openaiCompatible[id] = {
        ...(provider.api !== undefined && { api_url: provider.api }),
        available_models: availableModels,
      };
    }

    // `language_models` also holds first-party providers (openai, anthropic,
    // ...) managed through Zed's UI, so only the `openai_compatible` subtree
    // is recomputed; every other sibling rides along untouched.
    const existing = ZedConfigSchema.safeParse(JSON.parse(fileContent || "{}"));
    const existingLanguageModels =
      existing.success && existing.data.language_models
        ? { ...existing.data.language_models, openai_compatible: openaiCompatible }
        : { openai_compatible: openaiCompatible };

    return new ZedModels({
      outputRoot,
      relativeDirPath: paths.relativeDirPath,
      relativeFilePath: paths.relativeFilePath,
      fileContent: applySharedConfigPatch({
        fileKey: sharedConfigFileKey(paths),
        feature: "models",
        existingContent: fileContent,
        patch: { language_models: existingLanguageModels },
        filePath: join(outputRoot, paths.relativeDirPath, paths.relativeFilePath),
        logger,
      }),
      validate,
      global,
    });
  }

  toRulesyncModels(): RulesyncModels {
    const providers: Record<string, RulesyncModelProvider> = {};
    for (const [id, provider] of Object.entries(
      this.json.language_models?.openai_compatible ?? {},
    )) {
      if (isPrototypePollutionKey(id)) continue;
      const models: Record<string, RulesyncModel> = {};
      for (const model of provider.available_models ?? []) {
        const converted = fromZedModel(model, model.name ?? "");
        if (converted) models[converted.id] = converted;
      }
      providers[id] = {
        ...(provider.api_url !== undefined && { api: provider.api_url }),
        models,
      };
    }
    return this.toRulesyncModelsDefault({
      fileContent: JSON.stringify({ providers }, null, 2),
    });
  }

  validate(): ValidationResult {
    const json = parseJsoncStrict(this.fileContent || "{}");
    const result = ZedConfigSchema.safeParse(json);
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
  }: ToolModelsForDeletionParams): ZedModels {
    return new ZedModels({
      outputRoot,
      relativeDirPath,
      relativeFilePath,
      fileContent: "{}",
      validate: false,
      global,
    });
  }
}
