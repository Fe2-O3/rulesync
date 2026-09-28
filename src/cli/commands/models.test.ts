import { join } from "node:path";

import { parse as parseJsonc } from "jsonc-parser";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { mergeDiscoveredModels } from "../../features/models/models-source-io.js";
import { createMockLogger } from "../../test-utils/mock-logger.js";
import { setupTestDirectory } from "../../test-utils/test-directories.js";
import { ensureDir, readFileContent, writeFileContent } from "../../utils/file.js";
import { modelsFetchCommand } from "./models.js";

function stubFetch(json: unknown, ok = true, status = 200): typeof fetch {
  return (async () => ({ ok, status, json: async () => json })) as unknown as typeof fetch;
}

describe("models fetch/discover", () => {
  let testDir: string;
  let cleanup: () => Promise<void>;

  beforeEach(async () => {
    ({ testDir, cleanup } = await setupTestDirectory());
    vi.spyOn(process, "cwd").mockReturnValue(testDir);
    await ensureDir(join(testDir, ".rulesync"));
  });

  afterEach(async () => {
    await cleanup();
    vi.restoreAllMocks();
  });

  it("fetches one provider block from models.dev into the source file", async () => {
    await writeFileContent(
      join(testDir, ".rulesync", "models.jsonc"),
      '// local providers\n{"providers": {}}',
    );
    await modelsFetchCommand(createMockLogger(), {
      provider: "openai",
      outputRoot: testDir,
      fetchFn: stubFetch({ providers: { openai: { name: "OpenAI", models: {} } } }),
    });
    const written = await readFileContent(join(testDir, ".rulesync", "models.jsonc"));
    expect(written).toContain("// local providers");
    expect(
      (parseJsonc(written) as { providers: { openai: { name: string } } }).providers.openai.name,
    ).toBe("OpenAI");
  });

  it("rejects unknown models.dev providers", async () => {
    await writeFileContent(join(testDir, ".rulesync", "models.jsonc"), '{"providers": {}}');
    await expect(
      modelsFetchCommand(createMockLogger(), {
        provider: "nope",
        outputRoot: testDir,
        fetchFn: stubFetch({ providers: {} }),
      }),
    ).rejects.toThrow(/no provider/);
  });

  it("mergeDiscoveredModels fills missing limits without touching set values", () => {
    expect(
      mergeDiscoveredModels({
        incoming: { a: { id: "a", limit: { context: 5 } } },
        existing: { a: { id: "a", name: "A", limit: { output: 1 } } },
      }),
    ).toEqual({ a: { id: "a", name: "A", limit: { context: 5, output: 1 } } });
  });
});
