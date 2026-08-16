const fs = require('fs');
const { parseSPM } = require('./spm.js');
const pieces = parseSPM(fs.readFileSync('tokenizer.model'));
const hf = JSON.parse(fs.readFileSync('tokenizer.json', 'utf8'));
const hfMerges = hf.model.merges;
const score = new Map(), pidx = new Map();
pieces.forEach((p, i) => { score.set(p.piece, p.score); pidx.set(p.piece, i); });

// 1. Is HF's merge list sorted by score(a+b) descending?
let viol = 0, firstV = -1;
const sc = hfMerges.map(([a, b]) => score.get(a + b));
for (let i = 1; i < sc.length; i++) if (sc[i] > sc[i - 1]) { viol++; if (firstV < 0) firstV = i; }
console.log('HF merges: score(a+b) descending violations =', viol, 'first at', firstV);
console.log('  scores at idx 0..5:', sc.slice(0, 6));
console.log('  scores at idx 6900..6906:', sc.slice(6900, 6907));

// 2. Is HF's merge list sorted by piece INDEX of (a+b) ascending?
const pi = hfMerges.map(([a, b]) => pidx.get(a + b));
let viol2 = 0, firstV2 = -1;
for (let i = 1; i < pi.length; i++) if (pi[i] < pi[i - 1]) { viol2++; if (firstV2 < 0) firstV2 = i; }
console.log('HF merges: pieceIndex(a+b) ascending violations =', viol2, 'first at', firstV2);
console.log('  pieceIdx at 0..8:', pi.slice(0, 9));

// 3. Are merges for the same result piece contiguous?
let groups = 0, seen = new Set(), nonContig = 0;
let prev = -1;
for (const x of pi) { if (x !== prev) { if (seen.has(x)) nonContig++; seen.add(x); groups++; prev = x; } }
console.log('distinct result-piece groups:', groups, 'unique results:', seen.size, 'non-contiguous re-entries:', nonContig);

// 4. Within a group, what orders the multiple splits? sample a group with >1 merge
const byRes = new Map();
hfMerges.forEach(([a, b], i) => { const r = a + b; if (!byRes.has(r)) byRes.set(r, []); byRes.get(r).push([a, b, i]); });
const multi = [...byRes.entries()].filter(([, v]) => v.length > 2).slice(0, 3);
for (const [r, v] of multi) {
  console.log(`\nresult ${JSON.stringify(r)} pieceIdx=${pidx.get(r)} splits (HF order):`);
  for (const [a, b, i] of v) console.log(`   #${i} ${JSON.stringify(a)}(${pidx.get(a)}) + ${JSON.stringify(b)}(${pidx.get(b)})`);
}
