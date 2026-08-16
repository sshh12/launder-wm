# Launder WM

You are given a passage of real watermarked text. Wash the watermark out: edit
until a real detector stops flagging it, in as few word changes as you can.

The needle above the text is a SynthID-style detector running in your browser on
every keystroke. Under the box, each word is tinted by how much watermark signal
it carries — and because a token's score depends on the words before it, an edit
cools the words *after* it too. That ripple is the skill: cut upstream of a hot
run, not downstream of it.

## It is a real watermark, and deliberately not a secure one

The passages were generated with the Tournament sampling scheme from
"Scalable watermarking for identifying large language model outputs"
(Dathathri et al., Nature 2024), as implemented in Hugging Face `transformers`.
The 30 watermark keys are the canonical **published** set — the same ones in
Google DeepMind's `synthid-text` repository and the Hub detector configs — and
they are shipped to the browser on purpose.

Publishing the keys destroys the watermark's security. That is the point: you
can point stock `SynthIDTextWatermarkDetector` at any passage in `data/passages/`
and get the number on screen back. A needle that phones home is
indistinguishable from a needle that lies. This demonstrates a **real**
watermark; it does not demonstrate a **secure** one, and nothing here should be
read as claiming otherwise.

## The campaign

Fifteen levels, played in order — clear one and the next unlocks. Later levels
add rules to the gate: a change budget, a phrase that must survive verbatim,
meaning clamped to close paraphrase. Progress is anonymous, kept in
`localStorage` and mirrored on the server against a session id; there are no
accounts. `?level=N` opens any level directly.

## Two systems, and they run at different times

* **The watermark detector is local, fast and free, so it runs live.** It is a
  TypeScript port of the Python detector, debounced on every keystroke. Both
  implementations are held to the same committed golden vectors, bit for bit, by
  a CI gate — the browser is a verifier that re-derives what it shows.
* **The LLM judge fires once, on submit, and is a gate, not a score.** It is
  asked whether the submission is natural prose that still makes the passage's
  claims. Pass or fail; nothing to optimize. It closes what a free textarea
  opens — `the the the` is not natural, keyword soup does not preserve meaning —
  and it deliberately allows honest wholesale paraphrase, which lowers the
  reading legitimately at a high edit distance.

## Layout

```
packages/core    the watermark, the detector, scoring, the gate checks, schemas
packages/serve   FastAPI: renders the page, /api/detect, /api/submit, the judge
packages/forge   authoring studio: generate, analyze, solve, triage, calibrate
web/             the browser client: TS detector, tokenizer, needle, mirror
data/            committed build output: passages, keys, thresholds, all config
.build-docs/     the design record — TECH_PLAN.md, ARCHITECTURE.md, CONCEPT.md
```

`forge` needs a CUDA GPU and the gated Gemma-3 weights and is never installed in
production; `data/` is what it produces, committed, and the server only reads.

## Running it

```sh
uv sync                    # one venv for core, serve and the tests
make web-install           # npm ci, once
make web-build             # the server renders the built page, not a template
make serve                 # http://127.0.0.1:8000
```

Defaults are chosen so this costs nothing and needs nothing: a local SQLite file
for the database and a fake judge that always passes. Set `JUDGE_PROVIDER` and a
key in `.env` for the real one.

For work on the UI alone, `make web-dev` runs the Vite dev server against the
checked-in fixture passage. It has no API behind it, so play through `make
serve`.

## More

`.build-docs/TECH_PLAN.md` is the full technical plan — the detector's
arithmetic, the calibration, the asset budgets, the build. `ARCHITECTURE.md`
next to it records the decisions and what was rejected.

MIT licensed; see `LICENSE`, and `NOTICE` for what is redistributed and under
which terms.

Made by [@ShrivuShankar](https://x.com/ShrivuShankar).
