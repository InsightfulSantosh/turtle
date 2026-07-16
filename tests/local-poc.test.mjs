import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const artifactUrl = new URL("../app/generated-data.json", import.meta.url);
const pageUrl = new URL("../app/page.tsx", import.meta.url);

test("keeps the local POC and fitted model contract intact", async () => {
  const [artifactText, pageSource] = await Promise.all([
    readFile(artifactUrl, "utf8"),
    readFile(pageUrl, "utf8"),
  ]);
  const artifact = JSON.parse(artifactText);

  assert.equal(artifact.meta.model.version, "2.2.0");
  assert.equal(artifact.meta.model.demandLibrary, "scikit-learn");
  assert.equal(artifact.meta.visionModel.modelId, "patrickjohncyh/fashion-clip");
  assert.equal(artifact.meta.model.topK, 3);
  assert.match(pageSource, /FashionCLIP retrieval \+ scikit-learn demand model/);
});
