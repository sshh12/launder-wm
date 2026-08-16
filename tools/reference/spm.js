// Minimal SentencePiece ModelProto reader: ModelProto{ 1: repeated SentencePiece }
// SentencePiece{ 1: string piece, 2: float score, 3: enum type }
const fs = require('fs');

function readVarint(b, p) { let r = 0, s = 0, x; do { x = b[p.o++]; r |= (x & 0x7f) * Math.pow(2, s); s += 7; } while (x & 0x80); return r; }

function parseSPM(buf) {
  const p = { o: 0 }, pieces = [];
  while (p.o < buf.length) {
    const key = readVarint(buf, p), field = key >> 3, wire = key & 7;
    if (field === 1 && wire === 2) {
      const len = readVarint(buf, p), end = p.o + len;
      let piece = null, score = 0, type = 1;
      while (p.o < end) {
        const k2 = readVarint(buf, p), f2 = k2 >> 3, w2 = k2 & 7;
        if (f2 === 1 && w2 === 2) { const l = readVarint(buf, p); piece = buf.toString('utf8', p.o, p.o + l); p.o += l; }
        else if (f2 === 2 && w2 === 5) { score = buf.readFloatLE(p.o); p.o += 4; }
        else if (f2 === 3 && w2 === 0) { type = readVarint(buf, p); }
        else if (w2 === 2) { const l = readVarint(buf, p); p.o += l; }
        else if (w2 === 0) readVarint(buf, p);
        else if (w2 === 5) p.o += 4;
        else if (w2 === 1) p.o += 8;
        else throw new Error('wire ' + w2);
      }
      pieces.push({ piece, score, type });
      p.o = end;
    } else if (wire === 2) { const l = readVarint(buf, p); p.o += l; }
    else if (wire === 0) readVarint(buf, p);
    else if (wire === 5) p.o += 4;
    else if (wire === 1) p.o += 8;
    else throw new Error('top wire ' + wire);
  }
  return pieces;
}

module.exports = { parseSPM };

if (require.main === module) {
  const pieces = parseSPM(fs.readFileSync('tokenizer.model'));
  console.log('pieces:', pieces.length);
  console.log('first 8:', JSON.stringify(pieces.slice(0, 8)));
  console.log('types histogram:', pieces.reduce((m, x) => (m[x.type] = (m[x.type] || 0) + 1, m), {}));
  // is score monotonically non-increasing with index?
  let viol = 0, firstViol = -1;
  for (let i = 1; i < pieces.length; i++) {
    if (pieces[i].score > pieces[i - 1].score) { viol++; if (firstViol < 0) firstViol = i; }
  }
  console.log('score-order violations:', viol, 'first at idx', firstViol);
  console.log('around first viol:', JSON.stringify(pieces.slice(Math.max(0, firstViol - 3), firstViol + 3)));
  // distinct scores
  const sc = new Set(pieces.map(x => x.score));
  console.log('distinct scores:', sc.size);
  // score vs -index?
  console.log('sample scores at 0,100,1000,100000,262143:', [0, 100, 1000, 100000, 262143].map(i => pieces[i] && pieces[i].score));
}
