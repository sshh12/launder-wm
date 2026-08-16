// Compare @huggingface/tokenizers (JS) against Python/Rust `tokenizers` golden vectors,
// using BOTH the original tokenizer.json AND our reconstructed compact blob.
const fs = require('fs'), zlib = require('zlib');
const { Tokenizer } = require('./pkg/hft/package/dist/tokenizers.cjs');
const golden = JSON.parse(fs.readFileSync('golden.json', 'utf8'));
const hfJson = JSON.parse(fs.readFileSync('tokenizer.json', 'utf8'));
const hfCfg = JSON.parse(fs.readFileSync('tokenizer_config.json', 'utf8'));

const minimalCfg = { add_bos_token: true, add_eos_token: false, clean_up_tokenization_spaces: false, bos_token: '<bos>', eos_token: '<eos>', unk_token: '<unk>', pad_token: '<pad>', spaces_between_special_tokens: false, tokenizer_class: 'GemmaTokenizer' };

// rebuild from blob (same reader as e2e.js)
const { rebuildFromBlob } = require('./blobio.js');
const rebuilt = rebuildFromBlob(zlib.brotliDecompressSync(fs.readFileSync('gemma3-tok.bin.br')), hfJson.post_processor);

const tokOrig = new Tokenizer(hfJson, hfCfg);
const tokBlob = new Tokenizer(rebuilt, minimalCfg);

function run(name, tok) {
  let pass = 0, fail = 0; const examples = [];
  for (const g of golden) {
    const ids = tok.encode(g.text).ids;
    if (ids.length === g.ids.length && ids.every((x, i) => x === g.ids[i])) pass++;
    else { fail++; if (examples.length < 5) examples.push({ text: g.text.slice(0, 50), py: g.ids.slice(0, 15), js: ids.slice(0, 15) }); }
  }
  console.log(`${name}: ${pass}/${golden.length} match Python  (${fail} fail)`);
  for (const e of examples) console.log('   MISMATCH', JSON.stringify(e.text), '\n     py=', e.py, '\n     js=', e.js);
  return fail;
}

console.log('golden vectors:', golden.length);
const f1 = run('JS + original tokenizer.json ', tokOrig);
const f2 = run('JS + reconstructed 1.14MiB blob', tokBlob);

// perf on the real passage
const passage = golden[0].text;
const N = 500;
let t = process.hrtime.bigint();
for (let i = 0; i < N; i++) tokBlob.encode(passage);
console.log(`\nencode ${passage.split(/\s+/).length}-word passage x${N}: ${(Number(process.hrtime.bigint() - t) / 1e6 / N).toFixed(3)} ms each`);
console.log('EXIT', (f1 === 0 && f2 === 0) ? 'ALL PARITY OK' : 'PARITY FAILURE');
