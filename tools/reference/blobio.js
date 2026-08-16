// Reader for the compact Gemma-3 tokenizer blob.
class R {
  constructor(b) { this.b = b; this.o = 0; }
  v() { let r = 0, s = 0, x; do { x = this.b[this.o++]; r += (x & 0x7f) * Math.pow(2, s); s += 7; } while (x & 0x80); return r; }
  sv() { const u = this.v(); return (u & 1) ? -((u + 1) / 2) : u / 2; }
  u8() { return this.b[this.o++]; }
  str(n) { const s = this.b.toString('utf8', this.o, this.o + n); this.o += n; return s; }
}

function rebuildFromBlob(buf, postProcessor) {
  const r = new R(buf);
  const n = r.v();
  const vocabArr = new Array(n);
  for (let i = 0; i < n; i++) vocabArr[i] = r.str(r.v());
  const types = new Uint8Array(n).fill(1);
  const nnCount = r.v(); { let prev = 0; for (let i = 0; i < nnCount; i++) { prev += r.v(); types[prev] = r.u8(); } }

  const vocab = {}; const inVocab = new Set(vocabArr);
  for (let i = 0; i < n; i++) vocab[vocabArr[i]] = i;

  // canonical candidate derivation: piece index asc, split index asc
  const cand = [];
  for (let i = 0; i < n; i++) {
    const cps = Array.from(vocabArr[i]);
    for (let s = 1; s < cps.length; s++) {
      const l = cps.slice(0, s).join(''), rr = cps.slice(s).join('');
      if (inVocab.has(l) && inVocab.has(rr)) cand.push([l, rr]);
    }
  }
  const mLen = r.v();
  const merges = new Array(mLen);
  for (let i = 0; i < mLen; i++) merges[i] = cand[i + r.sv()];

  const aLen = r.v(); const added_tokens = new Array(aLen);
  { let prev = 0; for (let i = 0; i < aLen; i++) {
      prev += r.v(); const f = r.u8();
      const content = (f & 32) ? r.str(r.v()) : vocabArr[prev];
      added_tokens[i] = { id: prev, content, single_word: !!(f & 1), lstrip: !!(f & 2), rstrip: !!(f & 4), normalized: !!(f & 8), special: !!(f & 16) };
  } }

  return {
    version: '1.0', truncation: null, padding: null, added_tokens,
    normalizer: { type: 'Replace', pattern: { String: ' ' }, content: '▁' },
    pre_tokenizer: { type: 'Split', pattern: { String: ' ' }, behavior: 'MergedWithPrevious', invert: false },
    post_processor: postProcessor,
    decoder: { type: 'Sequence', decoders: [{ type: 'Replace', pattern: { String: '▁' }, content: ' ' }, { type: 'ByteFallback' }, { type: 'Fuse' }] },
    model: { type: 'BPE', dropout: null, unk_token: '<unk>', continuing_subword_prefix: null, end_of_word_suffix: null, fuse_unk: true, byte_fallback: true, ignore_merges: false, vocab, merges },
  };
}
module.exports = { rebuildFromBlob };
