import { join } from "node:path";

import { load } from "js-yaml";
import { describe, expect, it } from "vitest";

import { ModelsProcessor } from "../features/models/models-processor.js";
import { readFileContent, writeFileContent } from "../utils/file.js";
import { getHermesagentGlobalDir } from "../utils/hermesagent.js";
import {
  assertGenerateMatrixCoversTargets,
  runGenerate,
  runImport,
  useGlobalTestDirectories,
  useTestDirectory,
} from "./e2e-helper.js";

describe("e2e: models", () => {
  const { getTestDir } = useTestDirectory();

  it("generates opencode provider config from models.jsonc", async () => {
    const testDir = getTestDir();
    await writeFileContent(
      join(testDir, ".rulesync", "models.jsonc"),
      JSON.stringify({
        providers: {
          local: {
            api: "http://127.0.0.1:5678/v1",
            env: ["LOCAL_MODELS_KEY"],
            models: {
              "ollama/qwen3-coder:30b": {
                id: "ollama/qwen3-coder:30b",
                limit: { context: 131072, output: 16384 },
                tool_call: true,
              },
            },
          },
        },
        default: { provider: "local", model: "ollama/qwen3-coder:30b" },
      }),
    );

    await runGenerate({ target: "opencode", features: "models" });

    const written = JSON.parse(await readFileContent(join(testDir, "opencode.jsonc")));
    expect(written.model).toBe("local/ollama/qwen3-coder:30b");
    expect(written.provider.local.options).toEqual({
      baseURL: "http://127.0.0.1:5678/v1",
      apiKey: "{env:LOCAL_MODELS_KEY}",
    });
    expect(written.provider.local.models["ollama/qwen3-coder:30b"].limit).toEqual({
      context: 131072,
      output: 16384,
    });
  });

  it("imports opencode provider config back into models.jsonc", async () => {
    const testDir = getTestDir();
    await writeFileContent(
      join(testDir, "opencode.json"),
      JSON.stringify({
        provider: {
          local: {
            options: { baseURL: "http://127.0.0.1:5678/v1", apiKey: "{env:LOCAL_MODELS_KEY}" },
            models: {
              "ollama/qwen3-coder:30b": { limit: { context: 131072, output: 16384 } },
            },
          },
        },
        model: "local/ollama/qwen3-coder:30b",
      }),
    );

    await runImport({ target: "opencode", features: "models" });

    const imported = JSON.parse(await readFileContent(join(testDir, ".rulesync", "models.jsonc")));
    expect(imported.providers.local.api).toBe("http://127.0.0.1:5678/v1");
    expect(imported.providers.local.env).toEqual(["LOCAL_MODELS_KEY"]);
    expect(imported.default).toEqual({ provider: "local", model: "ollama/qwen3-coder:30b" });
  });

  it("generates Crush providers from models.jsonc", async () => {
    const testDir = getTestDir();
    await writeFileContent(
      join(testDir, ".rulesync", "models.jsonc"),
      JSON.stringify({
        providers: {
          local: {
            api: "http://127.0.0.1:5678/v1",
            env: ["LOCAL_MODELS_KEY"],
            models: {
              "ollama/qwen3-coder:30b": {
                id: "ollama/qwen3-coder:30b",
                limit: { context: 131072 },
              },
            },
          },
        },
        default: { provider: "local", model: "ollama/qwen3-coder:30b" },
      }),
    );

    await runGenerate({ target: "crush", features: "models" });

    const written = JSON.parse(await readFileContent(join(testDir, "crush.json")));
    expect(written.providers.local.api_key).toBe("$LOCAL_MODELS_KEY");
    expect(written.models.large).toEqual({ provider: "local", model: "ollama/qwen3-coder:30b" });
  });

  it("generates Zed providers from models.jsonc", async () => {
    const testDir = getTestDir();
    await writeFileContent(
      join(testDir, ".rulesync", "models.jsonc"),
      JSON.stringify({
        providers: {
          local: {
            api: "http://127.0.0.1:5678/v1",
            models: {
              "ollama/qwen3-coder:30b": {
                id: "ollama/qwen3-coder:30b",
                limit: { context: 131072 },
              },
            },
          },
        },
      }),
    );

    await runGenerate({ target: "zed", features: "models" });

    const written = JSON.parse(await readFileContent(join(testDir, ".zed", "settings.json")));
    expect(written.language_models.openai_compatible.local.available_models[0]).toMatchObject({
      name: "ollama/qwen3-coder:30b",
      max_tokens: 131072,
    });
  });

  it("generates Continue entries from models.jsonc", async () => {
    const testDir = getTestDir();
    await writeFileContent(
      join(testDir, ".rulesync", "models.jsonc"),
      JSON.stringify({
        providers: {
          local: {
            api: "http://127.0.0.1:5678/v1",
            models: {
              "ollama/qwen3-coder:30b": {
                id: "ollama/qwen3-coder:30b",
                limit: { context: 131072 },
                tool_call: true,
              },
            },
          },
        },
      }),
    );

    await runGenerate({ target: "continue", features: "models" });

    const written = load(
      await readFileContent(join(testDir, ".continue", "config.yaml")),
    ) as Record<string, unknown>;
    const entries = written.models as Record<string, unknown>[];
    expect(entries[0]).toMatchObject({
      name: "local/ollama/qwen3-coder:30b",
      capabilities: ["tool_use"],
    });
  });

  it("generates Qwen provider entries from models.jsonc", async () => {
    const testDir = getTestDir();
    await writeFileContent(
      join(testDir, ".rulesync", "models.jsonc"),
      JSON.stringify({
        providers: {
          local: {
            api: "http://127.0.0.1:5678/v1",
            env: ["LOCAL_MODELS_KEY"],
            models: {
              "ollama/qwen3-coder:30b": {
                id: "ollama/qwen3-coder:30b",
                limit: { context: 131072 },
              },
            },
          },
        },
      }),
    );

    await runGenerate({ target: "qwencode", features: "models" });

    const written = JSON.parse(await readFileContent(join(testDir, ".qwen", "settings.json")));
    expect(written.modelProviders.local[0]).toMatchObject({
      id: "ollama/qwen3-coder:30b",
      envKey: "LOCAL_MODELS_KEY",
    });
    expect(written.providerProtocol).toEqual({ local: "openai" });
  });

  it("covers every models target in the generate matrix", () => {
    // hermesagent, goose, and pi are global-only, so they are covered by
    // the global tests below rather than this project-scope matrix.
    assertGenerateMatrixCoversTargets({
      processor: ModelsProcessor,
      testedTargets: ["opencode", "crush", "zed", "continue", "qwencode"],
    });
  });
});

describe("e2e: models global", () => {
  const { getProjectDir, getHomeDir } = useGlobalTestDirectories();

  it("generates Hermes model config into the home directory", async () => {
    const projectDir = getProjectDir();
    const homeDir = getHomeDir();
    await writeFileContent(
      join(projectDir, ".rulesync", "models.jsonc"),
      JSON.stringify({
        providers: {
          local: {
            api: "http://127.0.0.1:5678/v1",
            env: ["LOCAL_MODELS_KEY"],
            models: { "ollama/qwen3-coder:30b": { id: "ollama/qwen3-coder:30b" } },
          },
        },
        default: { provider: "local", model: "ollama/qwen3-coder:30b" },
      }),
    );

    await runGenerate({
      target: "hermesagent",
      features: "models",
      global: true,
      env: { HOME_DIR: homeDir },
    });

    const written = load(
      await readFileContent(join(homeDir, getHermesagentGlobalDir(), "config.yaml")),
    ) as Record<string, unknown>;
    const model = written.model as Record<string, unknown>;
    expect(model.default).toBe("local/ollama/qwen3-coder:30b");
    expect(model.base_url).toBe("http://127.0.0.1:5678/v1");
    const providers = written.custom_providers as Record<string, unknown>[];
    expect(providers).toHaveLength(1);
    expect(providers[0]).toMatchObject({ base_url: "http://127.0.0.1:5678/v1" });
  });

  it("generates Goose provider files into the home directory", async () => {
    const projectDir = getProjectDir();
    const homeDir = getHomeDir();
    await writeFileContent(
      join(projectDir, ".rulesync", "models.jsonc"),
      JSON.stringify({
        providers: {
          local: {
            api: "http://127.0.0.1:5678/v1",
            env: ["LOCAL_MODELS_KEY"],
            models: {
              "ollama/qwen3-coder:30b": { id: "ollama/qwen3-coder:30b", limit: { context: 1 } },
            },
          },
        },
        default: { provider: "local", model: "ollama/qwen3-coder:30b" },
      }),
    );

    await runGenerate({
      target: "goose",
      features: "models",
      global: true,
      env: { HOME_DIR: homeDir },
    });

    const provider = JSON.parse(
      await readFileContent(join(homeDir, ".config", "goose", "custom_providers", "local.json")),
    );
    expect(provider).toMatchObject({ name: "local", api_key_env: "LOCAL_MODELS_KEY" });
    const config = load(
      await readFileContent(join(homeDir, ".config", "goose", "config.yaml")),
    ) as Record<string, unknown>;
    expect(config.active_provider).toBe("local");
  });

  it("generates pi providers into the home directory", async () => {
    const projectDir = getProjectDir();
    const homeDir = getHomeDir();
    await writeFileContent(
      join(projectDir, ".rulesync", "models.jsonc"),
      JSON.stringify({
        providers: {
          local: {
            api: "http://127.0.0.1:5678/v1",
            env: ["LOCAL_MODELS_KEY"],
            models: {
              "ollama/qwen3-coder:30b": { id: "ollama/qwen3-coder:30b", limit: { context: 1 } },
            },
          },
        },
      }),
    );

    await runGenerate({
      target: "pi",
      features: "models",
      global: true,
      env: { HOME_DIR: homeDir },
    });

    const written = JSON.parse(await readFileContent(join(homeDir, ".pi", "agent", "models.json")));
    expect(written.providers.local).toMatchObject({
      baseUrl: "http://127.0.0.1:5678/v1",
      apiKey: "$LOCAL_MODELS_KEY",
    });
  });
});
