// Assemble a layered PSD from raw full-canvas RGBA layers using ag-psd (same lib cv.cm ships).
// Usage: node assemble_psd.mjs <manifest.json> <out.psd>
import { readFileSync, writeFileSync } from "node:fs";
import { writePsd } from "ag-psd";

const [manifestPath, outPath] = process.argv.slice(2);
const man = JSON.parse(readFileSync(manifestPath, "utf8"));

const children = man.layers.map((l) => ({
  name: l.name,
  left: l.left || 0,
  top: l.top || 0,
  imageData: { data: new Uint8ClampedArray(readFileSync(l.file)), width: l.w, height: l.h },
}));

const out = writePsd({ width: man.width, height: man.height, children }, { generateThumbnail: false });
writeFileSync(outPath, Buffer.from(out));
console.error(`wrote ${outPath} (${(out.byteLength / 1048576).toFixed(1)} MB, ${children.length} layers)`);
