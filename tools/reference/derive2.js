const fs = require('fs');
const { parseSPM } = require('./spm.js');
const pieces = parseSPM(fs.readFileSync('tokenizer.model'));
const hf = JSON.parse(fs.readFileSync('tokenizer.json', 'utf8'));
const hfMerges = hf.model.merges;

const score = new Map();
for (const p of pieces) score.set(p.piece, p.score);

// Where do the score-0 whitespace pieces live in proto order vs HF merge order?
const ws = pieces.map((p, i) => [i, p.piece, p.score, p.type]).filter(x => /^[\n\t▁]+$/.test(x[1]));
console.log('whitespace-run pieces in proto order (first 10):', JSON.stringify(ws.slice(0, 10).map(x => [x[0], JSON.stringify(x[1]).slice(0, 14), x[2], x[3]])));
console.log('count:', ws.length);

// Build local lists once
const local_all = [];
for (let pi = 0; pi < pieces.length; pi++) {
  const merge = pieces[pi].piece, ps = pieces[pi].score;
  const cps = Array.from(merge);
  const local = [];
  for (let index = 1; index < cps.length; index++) {
    const l = cps.slice(0, index).join(''), r = cps.slice(index).join('');
    if (score.has(l) && score.has(r)) local.push([l, r, ps, pi]);
  }
  local.sort((a, b) => (score.get(b[0]) - score.get(a[0])) || (score.get(b[1]) - score.get(a[1])));
  for (const x of local) local_all.push(x);
}
console.log('total candidate merges:', local_all.length);

function check(name, cmp) {
  const idx = local_all.map((_, i) => i);
  idx.sort(cmp);
  let mism = 0, first = -1;
  for (let i = 0; i < idx.length; i++) {
    const d = local_all[idx[i]];
    if (d[0] !== hfMerges[i][0] || d[1] !== hfMerges[i][1]) { mism++; if (first < 0) first = i; }
  }
  console.log(`${name.padEnd(46)} mismatches=${mism} first=${first}`);
  return mism;
}

check('score desc, tie: original order (stable)', (a, b) => (local_all[b][2] - local_all[a][2]) || (a - b));
check('score desc, tie: reverse original order', (a, b) => (local_all[b][2] - local_all[a][2]) || (b - a));
check('score desc, tie: piece index desc', (a, b) => (local_all[b][2] - local_all[a][2]) || (local_all[b][3] - local_all[a][3]) || (a - b));
check('score desc, tie: merged len desc', (a, b) => (local_all[b][2] - local_all[a][2]) || ((local_all[b][0].length + local_all[b][1].length) - (local_all[a][0].length + local_all[a][1].length)) || (a - b));
