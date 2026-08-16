const fs = require('fs'), zlib = require('zlib');
const t = JSON.parse(fs.readFileSync('tokenizer.json', 'utf8'));
const vocab = t.model.vocab, merges = t.model.merges;

const N = 262144;
const strById = new Array(N);
const idByStr = new Map();
for (const [s, i] of Object.entries(vocab)) { strById[i] = s; idByStr.set(s, i); }

const br = (buf, q = 11) => zlib.brotliCompressSync(buf, {
  params: {
    [zlib.constants.BROTLI_PARAM_QUALITY]: q,
    [zlib.constants.BROTLI_PARAM_LGWIN]: 24,
    [zlib.constants.BROTLI_PARAM_SIZE_HINT]: buf.length,
  }
});
const gz = (buf) => zlib.gzipSync(buf, { level: 9 });
const kb = n => (n / 1024).toFixed(0) + ' KiB';
const mb = n => (n / 1048576).toFixed(2) + ' MiB';

function report(name, buf) {
  const b = br(buf), g = gz(buf);
  console.log(`${name.padEnd(34)} raw=${String(buf.length).padStart(9)} (${mb(buf.length).padStart(9)})  br=${String(b.length).padStart(8)} (${mb(b.length)})  gz=${String(g.length).padStart(8)}`);
  return b.length;
}

// ---- varint helper
class W {
  constructor() { this.a = []; }
  v(n) { while (n >= 0x80) { this.a.push((n & 0x7f) | 0x80); n >>>= 7; } this.a.push(n); }
  u8(n) { this.a.push(n & 0xff); }
  bytes(b) { for (const x of b) this.a.push(x); }
  buf() { return Buffer.from(this.a); }
}

console.log('=== BASELINES ===');
report('tokenizer.json (raw HF)', fs.readFileSync('tokenizer.json'));
report('tokenizer.model (SP proto)', fs.readFileSync('tokenizer.model'));

// ---- (1) naive: all vocab strings, id order, length-prefixed + merges as id pairs varint
console.log('\n=== COMPONENTS ===');
{
  const w = new W();
  for (let i = 0; i < N; i++) { const b = Buffer.from(strById[i], 'utf8'); w.v(b.length); w.bytes(b); }
  report('A. all vocab strings (len+utf8)', w.buf());
}

// ---- (2) vocab as literal-or-pair (back-reference) in id order
// tag byte: 0 = pair(varint a, varint b); n>0 = literal of length n
{
  // pick, for each id that is a merge result, the FIRST merge that produces it
  const producer = new Array(N).fill(null);
  for (const [a, b] of merges) {
    const id = idByStr.get(a + b);
    if (producer[id] === null) producer[id] = [idByStr.get(a), idByStr.get(b)];
  }
  let lits = 0, pairs = 0, badOrder = 0;
  const w = new W();
  for (let i = 0; i < N; i++) {
    const p = producer[i];
    // only usable as a back-ref if both components have SMALLER id (decodable in one pass)
    if (p && p[0] < i && p[1] < i) { w.u8(0); w.v(p[0]); w.v(p[1]); pairs++; }
    else {
      if (p) badOrder++;
      const b = Buffer.from(strById[i], 'utf8');
      if (b.length > 250) { w.u8(255); w.v(b.length); } else w.u8(b.length);
      w.bytes(b); lits++;
    }
  }
  console.log(`   (back-ref pairs=${pairs}, literals=${lits}, forward-ref fallbacks=${badOrder})`);
  report('B. vocab as literal-or-backref', w.buf());
}

// ---- (3) merge table: varint pairs, rank order
{
  const w = new W();
  for (const [a, b] of merges) { w.v(idByStr.get(a)); w.v(idByStr.get(b)); }
  report('C. merges varint (a,b) rank order', w.buf());
}

// ---- (4) merge table: fixed 24-bit LE, interleaved
{
  const buf = Buffer.alloc(merges.length * 6);
  let o = 0;
  for (const [a, b] of merges) {
    const x = idByStr.get(a), y = idByStr.get(b);
    buf[o++] = x & 255; buf[o++] = (x >> 8) & 255; buf[o++] = (x >> 16) & 255;
    buf[o++] = y & 255; buf[o++] = (y >> 8) & 255; buf[o++] = (y >> 16) & 255;
  }
  report('D. merges u24 interleaved', buf);
}

// ---- (5) merge table: byte-plane transposed (struct-of-arrays) — helps brotli a lot
{
  const M = merges.length;
  const buf = Buffer.alloc(M * 6);
  let k = 0;
  const A = new Int32Array(M), B = new Int32Array(M);
  for (let i = 0; i < M; i++) { A[i] = idByStr.get(merges[i][0]); B[i] = idByStr.get(merges[i][1]); }
  for (const arr of [A, B]) for (let sh = 0; sh < 24; sh += 8) for (let i = 0; i < M; i++) buf[k++] = (arr[i] >> sh) & 255;
  report('E. merges u24 byte-plane transposed', buf);
}

// ---- (6) merges sorted by (a,b) with delta coding on `a` + rank stored separately
{
  const M = merges.length;
  const rec = new Array(M);
  for (let i = 0; i < M; i++) rec[i] = [idByStr.get(merges[i][0]), idByStr.get(merges[i][1]), i];
  rec.sort((p, q) => p[0] - q[0] || p[1] - q[1]);
  const w = new W();
  let prevA = 0;
  for (const [a, b, r] of rec) { w.v(a - prevA); prevA = a; w.v(b); w.v(r); }
  report('F. merges sorted+delta a, +rank', w.buf());
}

// ---- (7) FULL BUNDLE: back-ref vocab + transposed merges, single brotli stream
{
  const producer = new Array(N).fill(null);
  for (const [a, b] of merges) {
    const id = idByStr.get(a + b);
    if (producer[id] === null) producer[id] = [idByStr.get(a), idByStr.get(b)];
  }
  const w = new W();
  for (let i = 0; i < N; i++) {
    const p = producer[i];
    if (p && p[0] < i && p[1] < i) { w.u8(0); w.v(p[0]); w.v(p[1]); }
    else { const b = Buffer.from(strById[i], 'utf8'); if (b.length > 250) { w.u8(255); w.v(b.length); } else w.u8(b.length); w.bytes(b); }
  }
  const vocabBuf = w.buf();

  const M = merges.length;
  const mbuf = Buffer.alloc(M * 6); let k = 0;
  const A = new Int32Array(M), B = new Int32Array(M);
  for (let i = 0; i < M; i++) { A[i] = idByStr.get(merges[i][0]); B[i] = idByStr.get(merges[i][1]); }
  for (const arr of [A, B]) for (let sh = 0; sh < 24; sh += 8) for (let i = 0; i < M; i++) mbuf[k++] = (arr[i] >> sh) & 255;

  const bundle = Buffer.concat([vocabBuf, mbuf]);
  console.log('');
  console.log('=== FULL BUNDLE (vocab backref + merges transposed) ===');
  report('G. BUNDLE single stream', bundle);
  console.log(`   (separately: vocab br=${br(vocabBuf).length}, merges br=${br(mbuf).length}, sum=${br(vocabBuf).length + br(mbuf).length})`);
}
