import { z } from "zod/mini";

import { RULESYNC_MODELS_RELATIVE_FILE_PATH } from "../../constants/rulesync-paths.js";
import { FeatureProcessor } from "../../types/feature-processor.js";
import { RulesyncFile } from "../../types/rulesync-file.js";
import { ToolFile } from "../../types/tool-file.js";
import { modelsProcessorToolTargetTuple } from "../../types/tool-target-tuples.js";
import { ToolTarget } from "../../types/tool-targets.js";
import { formatError } from "../../utils/error.js";
import type { Logger } from "../../utils/logger.js";
import { ContinueModels } from "./continue-models.js";
import { CrushModels } from "./crush-models.js";
import { HermesModels } from "./hermesagent-models.js";
import { OpencodeModels } from "./opencode-models.js";
import { RulesyncModels } from "./rulesync-models.js";
import {
  ToolModels,
  ToolModelsForDeletionParams,
  ToolModelsFromFileParams,
  ToolModelsFromRulesyncModelsParams,
  ToolModelsSettablePaths,
} from "./tool-models.js";
import { ZedModels } from "./zed-models.js";

export type ModelsProcessorToolTarget = (typeof modelsProcessorToolTargetTuple)[number];

export const ModelsProcessorToolTargetSchema = z.enum(modelsProcessorToolTargetTuple);

type ToolModelsFactory = {
  class: {
    fromRulesyncModels(
      params: ToolModelsFromRulesyncModelsParams & { global?: boolean },
    ): ToolModels | Promise<ToolModels>;
    fromFile(params: ToolModelsFromFileParams): Promise<ToolModels>;
    forDeletion(params: ToolModelsForDeletionParams): ToolModels;
    getSettablePaths(options?: { global?: boolean }): ToolModelsSettablePaths;
  };
  meta: {
    supportsProject: boolean;
    supportsGlobal: boolean;
  };
};

export const toolModelsFactories = new Map<ModelsProcessorToolTarget, ToolModelsFactory>([
  ["opencode", { class: OpencodeModels, meta: { supportsProject: true, supportsGlobal: true } }],
  [
    "hermesagent",
    {
      // Global-only, like Hermes MCP: no project-scoped models location exists.
      class: HermesModels,
      meta: { supportsProject: false, supportsGlobal: true },
    },
  ],
  [
    "crush",
    {
      class: CrushModels,
      meta: { supportsProject: true, supportsGlobal: true },
    },
  ],
  [
    "zed",
    {
      class: ZedModels,
      meta: { supportsProject: true, supportsGlobal: true },
    },
  ],
  [
    "continue",
    {
      class: ContinueModels,
      meta: { supportsProject: true, supportsGlobal: true },
    },
  ],
]);

const allToolTargetKeys = [...toolModelsFactories.keys()];

const modelsProcessorToolTargets: ToolTarget[] = allToolTargetKeys.filter(
  (target) => toolModelsFactories.get(target)?.meta.supportsProject ?? false,
);

const modelsProcessorToolTargetsGlobal: ToolTarget[] = allToolTargetKeys.filter(
  (target) => toolModelsFactories.get(target)?.meta.supportsGlobal ?? false,
);

type GetFactory = (target: ModelsProcessorToolTarget) => ToolModelsFactory;

const defaultGetFactory: GetFactory = (target) => {
  const factory = toolModelsFactories.get(target);
  if (!factory) {
    throw new Error(`Unsupported tool target: ${target}`);
  }
  return factory;
};

export class ModelsProcessor extends FeatureProcessor {
  private readonly toolTarget: ModelsProcessorToolTarget;
  private readonly global: boolean;

  constructor({
    outputRoot = process.cwd(),
    inputRoots,
    toolTarget,
    global = false,
    dryRun = false,
    logger,
  }: {
    outputRoot?: string;
    inputRoots?: readonly [string, ...string[]] | readonly string[];
    toolTarget: ToolTarget;
    global?: boolean;
    dryRun?: boolean;
    logger: Logger;
  }) {
    super({ outputRoot, inputRoots, dryRun, logger });
    const result = ModelsProcessorToolTargetSchema.safeParse(toolTarget);
    if (!result.success) {
      throw new Error(
        `Invalid tool target for ModelsProcessor: ${toolTarget}. ${formatError(result.error)}`,
      );
    }
    this.toolTarget = result.data;
    this.global = global;
  }

  private get factory(): ToolModelsFactory {
    return defaultGetFactory(this.toolTarget);
  }

  /** Absence is ordinary: the feature does nothing without its source file. */
  async loadRulesyncFiles(): Promise<RulesyncFile[]> {
    try {
      return [await RulesyncModels.fromRoots({ inputRoots: this.inputRoots, logger: this.logger })];
    } catch (error) {
      this.reportRulesyncSourceLoadError({
        message: `Failed to load a Rulesync models file (${RULESYNC_MODELS_RELATIVE_FILE_PATH})`,
        error,
      });
      return [];
    }
  }

  async loadToolFiles({
    forDeletion = false,
  }: {
    forDeletion?: boolean;
  } = {}): Promise<ToolFile[]> {
    const paths = this.factory.class.getSettablePaths({ global: this.global });

    if (forDeletion) {
      const toolModels = this.factory.class.forDeletion({
        outputRoot: this.outputRoot,
        relativeDirPath: paths.relativeDirPath,
        relativeFilePath: paths.relativeFilePath,
        global: this.global,
      });
      return toolModels.isDeletable() ? [toolModels] : [];
    }

    return [
      await this.factory.class.fromFile({
        outputRoot: this.outputRoot,
        validate: true,
        global: this.global,
        logger: this.logger,
      }),
    ];
  }

  async convertRulesyncFilesToToolFiles(rulesyncFiles: RulesyncFile[]): Promise<ToolFile[]> {
    const rulesyncModels = rulesyncFiles.find(
      (file): file is RulesyncModels => file instanceof RulesyncModels,
    );

    if (!rulesyncModels) {
      throw new Error(`No ${RULESYNC_MODELS_RELATIVE_FILE_PATH} found.`);
    }

    return [
      await this.factory.class.fromRulesyncModels({
        outputRoot: this.outputRoot,
        rulesyncModels,
        global: this.global,
        logger: this.logger,
      }),
    ];
  }

  async convertToolFilesToRulesyncFiles(toolFiles: ToolFile[]): Promise<RulesyncFile[]> {
    const toolModelsList = toolFiles.filter(
      (file): file is ToolModels => file instanceof ToolModels,
    );

    const rulesyncModelsList = toolModelsList.map((toolModels) => {
      return toolModels.toRulesyncModels();
    });

    return rulesyncModelsList;
  }

  static getToolTargets({ global = false }: { global?: boolean } = {}): ToolTarget[] {
    if (global) {
      return modelsProcessorToolTargetsGlobal;
    }
    return modelsProcessorToolTargets;
  }
}
