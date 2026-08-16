import json, random, sys
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer.json")

cases = []
with open("passage.txt", encoding="utf-8") as f:
    cases.append(f.read().strip())

cases += [
    "Hello World", "", " ", "  ", "   leading spaces", "trailing   ", "a  b   c",
    "emoji 🧼🚀 and CJK 中文字符 and Arabic العربية",
    "zero​width​space", "soft­hyphen", "﻿BOM start",
    "NFKC ﬁ ligature ①②③ ｆｕｌｌｗｉｄｔｈ",
    "tabs\there\tand\nnewlines\n\n\nmany",
    "<bos> literal special <start_of_turn> tokens <unused0> <image_soft_token>",
    "combining é vs é", "ß ẞ Straße",
    "digits 1234567890 and 3.14159", "a" * 500, "  control", "ﷺ﷽ rare",
    "Ελληνικά Русский हिन्दी 日本語 한국어", "  \t \n mixed  ws \t\t ",
    "ȩ́ stacked marks", "\U0001F1FA\U0001F1F8 flag", "\U0001F469‍\U0001F4BB zwj",
    "line1\r\nline2\rline3", "\x00\x01 control bytes", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 math",
    "  ▁ literal metaspace ▁▁▁ ", "mixed▁and space",
]

# deterministic fuzz across unicode planes
rng = random.Random(20260815)
for i in range(4000):
    n = 1 + (i % 40)
    s = []
    for _ in range(n):
        r = rng.random()
        if r < 0.45:
            cp = rng.randint(32, 126)
        elif r < 0.65:
            cp = rng.randint(0xA0, 0x2FFF)
        elif r < 0.85:
            cp = rng.randint(0x4E00, 0x9FFF)
        elif r < 0.95:
            cp = rng.randint(0x0600, 0x06FF)
        else:
            cp = rng.randint(0x1F300, 0x1FAFF)
        s.append(chr(cp))
    cases.append("".join(s))

out = []
for c in cases:
    e = tok.encode(c)
    out.append({"text": c, "ids": e.ids, "tokens": e.tokens})

with open("golden.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)

print("wrote", len(out), "golden vectors")
print("passage ids len:", len(out[0]["ids"]))
print("first 12 ids:", out[0]["ids"][:12])
print("first 12 tokens:", out[0]["tokens"][:12])
