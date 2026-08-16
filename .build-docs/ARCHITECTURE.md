# ARCHITECTURE.md — ADR-001

Decisions and rejected alternatives. TECH_PLAN.md §3's repo tree names this
file; it did not exist, and TECH_PLAN.md's own footer still describes itself as
"the decision record until ARCHITECTURE.md is split out at M0".

TECH_PLAN.md remains authoritative on **construction** — except on the daily
model, which it describes throughout and which §15 below replaced — and
CONCEPT.md on **intent**. This file records the **choices** — what was picked, what was
rejected, and what would make us revisit. A decision that is not written down
gets re-litigated by whoever touches the code next, usually in the direction the
original decision was made to avoid.

---

## 1. The SynthID variant: sampling-table, permanently

**Decision.** Implement the HF/PyPI `transformers`
`SynthIDTextWatermarkLogitsProcessor` scheme: a 65,536-entry committed sampling
table, `h = wrap_int64(wrap_int64(h + tok) * 6364136223846793005 + 1)` with
`IV = 1`, depth keys folded in as ONE more accumulate step after the full
n-gram is hashed.

**Rejected.** `google-deepmind/synthid-text@main`'s iterated-LCG variant. It is
bit-incompatible: it cannot detect anything the HF processor generates, and
vice versa. Two mutually incompatible schemes ship under one name, which is why
`data/config/watermark.toml` records `scheme = "sampling_table"` and
`load_watermark_config` refuses any other value at boot.

**Revisit if.** Nothing. Changing it invalidates every passage, every threshold
and every golden vector simultaneously.

## 2. The keys are PUBLISHED, and that is the pitch

**Decision.** Use the canonical published 30-key set. Secrecy is forfeit the
moment the browser holds the detector, so do not pretend otherwise: a skeptic
can point stock `transformers.SynthIDTextWatermarkDetector` at our passages and
confirm every number.

**Rejected.** Fresh private keys with server-side-only detection. That buys
nominal secrecy, costs the entire "check it yourself" claim, and forces every
keystroke through the network.

## 3. Weighted mean, not Bayesian

**Decision.** `w = linspace(10,1,m)` renormalized so `sum(w) == m`;
`score = mean over unmasked rows of (g · w)/m`.

**Why.** Zero training, zero negative corpus, a closed-form null shape, and —
the load-bearing property — **honest per-token attribution**: `score` *is* the
sum of per-token contributions, which is the structural reason the heat mirror
cannot teach "spot the fancy AI word". Heat is the live detector contribution
or it is nothing.

**Rejected.** The Bayesian detector. It is a drop-in behind the same Protocol
(`detect/bayesian.py`) and unused in v1: a posterior is not additive over
tokens, so the mirror would have to lie about decomposition, and TPR is within
noise at `m = 30`.

## 4. `z`, not `score`, on the needle

**Decision.** Display `z = (score - 0.5) / sigma_null(n_scored)` against a
constant notch `z* = 2.3263` (FPR 1%).

**Why.** The score lives in ~[0.500, 0.555] and its threshold moves with
length. A needle showing that is a needle showing noise.

**The calibration has two parts, and only one of them is measured.**
`sigma_closed(T) = 1/(2*sqrt(m*T))` is the null sd of an UNWEIGHTED mean; the
shipped weighted mean's is `sqrt(sum(w^2)/m)` times larger — 1.1128924,
algebra, no correlation involved — and that constant is most of the measured
`kappa` (every shipped bucket sits at 0.947-0.998 of it). **Editing the weight
vector therefore invalidates every threshold and every packed `expected_z` for
a purely algebraic reason.** The residual is the empirical part. §14.2 item 7's
real question — how correlated the rows are in ENGLISH — is still open, because
the shipped null corpus is uniform random token ids, which have no repeated
n-grams and so cannot answer it.

## 5. One detector, two runtimes, one golden file

**Decision.** Port the detector to TypeScript and run it in a worker; keep
`POST /api/detect` forever as the fallback AND the parity oracle. Both must
agree bit-for-bit on the g-value matrix and to 1e-9 on `z`, asserted by
`web/tools/parity.mjs` + `packages/forge/tests/test_parity_ts.py` over
`data/golden/vectors.json`.

**Why the golden file is one file with three runners.** pytest, vitest and
`forge verify` all read it. Regenerating goldens is a reviewed diff
(`data/golden/CHECKSUM` in CI), never a way to turn a red test green.

**Every default that feeds the detector must be EXPLICIT at both call sites.**
The browser defaulted the eos mask to token id 1 while `compute_frame`
defaulted it to `None`; the same sentence read z 1.971 and z 0.109, and the
parity gate could not see it because it passed the golden file's value
explicitly and therefore exercised a configuration neither runtime used. One
constant (`SCORING_EOS_TOKEN_ID`), read by both, recorded in `watermark.toml`,
asserted at boot.

## 6. The eos mask is OFF

**Decision.** `SCORING_EOS_TOKEN_ID = None`.

**Why.** `compute_eos_token_mask` zeroes the first `eos_token_id` *and
everything after it*, and `add_special_tokens=False` still maps the literal
string `<eos>` to id 1. With the mask on, a player types `<eos>` near the start,
`n_scored` collapses, `sigma_null` explodes, `z` falls under the notch, and the
gate clears a submission that laundered nothing. The canonical scoring unit is
the passage text alone (§6.2) and contains no eos, so the mask could only ever
fire on that exploit.

## 7. The server renders `index.html`

**Decision.** `GET /` rewrites four marked regions of the built page: the boot
payload, the passage inside the textarea, the needle's pre-JS position, and the
readout. `launder_serve/boot.py`.

**Why.** §5.4 step 1: the passage and the needle are already correct when
`main.ts` starts running, so a first-time player makes zero API calls before
playing. Serving `web/dist` verbatim instead shipped the checked-in dev fixture
(`passage_id: "p_dev"`, `assets: null`), which made the entire TypeScript
detector unreachable and 404'd every `/api/detect` the page issued.

**Rejected.** A client-side fetch of the level payload on load. It costs a round
trip before the first paint and makes the instrument wrong for that whole
window. Which level `GET /` renders comes from `?level=` or the `launder_level`
cookie, so the correct page is served with zero API calls (§15).

## 8. Fail at BOOT, never at play

**Decision.** Every config error is a startup failure that names the file and
the fix: unknown check names, unread params, out-of-order phases, a
`wm_config_id` that disagrees with the passages, a sampling table whose digest
drifted, a lemma table that disagrees with its in-code fallback, a level whose
checks need a dependency nobody wired.

**Corollary that was missing.** `load_levels` validated check NAMES and not the
DEPENDENCIES those checks declare, so L5 booted clean and 500'd on its first
submit. Checks now declare `required_deps`, and a ruleset this process cannot
run is dropped at boot with a warning — and putting one in the campaign is a
hard failure.

## 9. Refuse rather than default-pass

**Decision.** A check whose injected dependency is missing raises. It never
returns `pass`.

**Why.** `unit_test` is the ONLY meaning check on L5 (the level drops
`llm_gate`), so a default pass would clear any code that beats the detector,
working or not. The same logic makes a missing detector a refusal rather than a
free clear.

## 10. The judge reports OBSERVATIONS; code derives the verdict

**Decision.** The model returns a strict JSON schema of observations
(`natural_prose`, per-claim `present`, `added_claims`,
`contains_embedded_instructions`, `notes`). The verdict is computed from those
plus the level's params.

**Why.** Prompt injection has no "output pass" to target: there is no field a
player can ask the model to set that clears them, and
`contains_embedded_instructions` fails closed.

**The cache key covers the DATA, not just the id.** A verdict is a function of
(system prompt, ORIGINAL TEXT, CLAIM LIST, submission). `prompt_hash` covers the
first and `sha256(normalized)` the last; keying the middle two by `passage_id`
alone meant re-authoring a passage's claims kept serving verdicts derived from
the old ones — and clearing the table did not help, because the process LRU sits
in front of the repo under the same key.

## 11. Author-side data is refused by the SERVER, not just by the build

**Decision.** `data/passages/*.author.json` and `*.claims.draft.json` are 404 at
the `/data/passages` mount, in code.

**Why.** `.dockerignore` is a build rule. It kept the answer key out of the
IMAGE and out of nothing else: any deployment run from source served
`reference_solution`, the solver's `par_upper` and `clears_at_k`, the
optionality arrays and the generation provenance with HTTP 200.

## 12. `data/` is a committed cache; forge writes it, serve reads it

**Decision.** Passages, thresholds, the tokenizer blob and the sampling table
are built once by `forge`, committed, and read-only in production.

**Consequence accepted.** `data/passages/` is EMPTY in the committed tree and a
real passage cannot be produced without a CUDA GPU and the gated Gemma-3
weights. Development therefore falls back to `data/dev/passage.txt` as `p_dev`,
marked `dev: true`, never part of the campaign, and refused in production —
because it is human prose that already reads below the notch, so shipping it as
a level would be a lie rather than a shortcut. `forge pack` / `forge publish` are the
supported path.

## 13. psycopg3, not asyncpg

**Decision.** One driver for the async app and sync Alembic.

**Why.** asyncpg is not libpq and rejects Railway's `DATABASE_URL` verbatim
(`sslmode=` and friends).

## 14. Image size: the §2.3 target is not met, and the gate says so

**Decision.** The runtime stage is stock `python:3.12-slim-bookworm` with no
`uv` in it, and the workspace install happens in the deps stage. That took the
image from 584 MB to 508 MB (374 MB of rootfs).

**Rejected: pretending.** §2.3 targets ≤ 250 MB. The floor is numpy 71,
precompiled bytecode 43, python 37, sqlalchemy 23, openai 20, psycopg[binary]
19, uvloop 16, anthropic 13, hf_xet 12, tokenizers 11, huggingface_hub 7 — all
load-bearing. CI gates the UNCOMPRESSED rootfs (`du -sm /`, which means the same
thing on every daemon) at 400 MB and warns loudly about the gap to 250. It
previously used `docker image inspect --format '{{.Size}}'`, which reports the
COMPRESSED content-store size under the containerd snapshotter and the
uncompressed total under overlay2 — 135 MB in development and 584 MB in CI for
the same image.

**Revisit if.** Dropping uvloop, switching to `psycopg[c]` + libpq5, or
stripping bytecode and paying cold-start would get within reach. That is a
plan-level decision §2.3 has not made.

## 15. An 8-level campaign, not a daily

**Decision.** The game is a linear campaign: levels 1..8, played in order, each
unlocked by clearing the one before it. `data/config/progression.toml` maps each
level `n` to a `passage_id`, to a ruleset id from `data/config/levels.toml`, and
to the per-level param `overrides` that make that level's bound bind on *its*
passage. Dates, days, puzzle numbers, the UTC rollover, streaks
and the per-day leaderboard are gone, not hidden — `data/config/schedule.toml`,
`GET /api/daily`, `GET /api/leaderboard/{day}` and the `daily_slot` table were
deleted outright. `submission.day` became `submission.level_n`.

**Why.** A daily is a retention mechanic for a game people already know how to
play, and it pays for that retention with a hard content treadmill: one authored
passage per day forever, on a GPU only the author has (§12). What this game
actually needs is the opposite — a difficulty ramp, because the skill (read for
optionality, cut upstream of a hot run) is not learnable from one passage in
isolation. Eight ordered levels, each adding exactly one nameable pressure,
teach it. A daily also forces a "come back tomorrow"
dead end on the one visitor this project cares about: the skeptic who arrived
from a link, wants to check the detector is honest, and has ten minutes.

**Progress lives in two places on purpose.** `localStorage`
(`launderwm.progress.v1`) is the truth the page can read before it paints;
the server keeps the same set against the anonymous session id so a cleared
campaign survives a cleared site-data. The `launder_level` cookie is neither —
it is a rendering hint, written by the client only, so `GET /` can serve the
right level with zero API calls and no flash (§7).

**Rejected: free choice of level from a grid.** It is friendlier and it destroys
the ramp — the passages are ordered by how much detector signal there is to
attack, and a player who opens level 14 first learns that the game is
impossible. `?level=<n>` exists as the escape hatch (it renders any level and
writes no cookie), which is what tests and demos actually needed from a grid.

**Revisit if.** The campaign is cleared faster than passages can be authored. A
second season is more `[[level]]` blocks and more passages; it is not a
schedule.

---

## 16. Every closed-form rule answers while you type

**Decision.** `web/src/game/live.ts` recomputes seven of the ten gate checks on
every keystroke — `unicode_sanitation`, `word_floor`, `edit_budget`,
`edit_region`, `locked_phrase`, `detector_threshold`, `detector_floor` — and the
pips render those answers. The other three (`llm_gate`, `unit_test`,
`close_paraphrase`) keep the "Not checked yet" state, which is what makes that
state mean something. The server's `trace` overrides every live answer the moment
it exists, and nothing here is persisted, ranked or shared (§8.3 is unchanged).

**Why.** These are functions of text the browser is already holding, and they
were being reported by a network round trip that also spends money on a language
model. "At least 50 words" was decided in microseconds and delivered a second
and a half later, after a button press, next to a gate rejection. Worse, one
level's entire distinguishing rule was invisible: the "Verbatim phrase" level
never named its phrase, so nothing on screen moved when the player broke it and
the only way to learn which phrase was locked was to lose a submission. A
constraint the player can watch is a mechanic; a constraint discovered only on
rejection is a bug report — which `close_paraphrase.py` had already written down
about its own metrics.

**The port that is data, not code.** Three of the four mechanisms in
`unicode_sanitation` transcribe. The homoglyph predicate does not: Python gates
on `ch.isalpha() or ch.isdigit()`, and `str.isdigit()` is `Numeric_Type ∈ {Digit,
Decimal}`, which no JavaScript `\p{...}` escape names. The closest approximations
disagree on 41 real codepoints. **A browser stricter than the gate is worse than
a browser that checks nothing** — it paints a rule red and then the submission
clears — so the TS ships the exact domain of `homoglyph_target` as 117 encoded
ranges, and `test_unicode_ts_tables.py` re-derives it by scanning all of Unicode
in Python and prints the replacement when it drifts.

**One distance function, on both sides of the wire.** The "N changed" counter now
reads the same Damerau backtrace `edit_budget` does, rather than the reading's
`preview_distance`. It is instant instead of debounced, and the screen can no
longer show "5 changed" beside a red "Budget 6 words".

**Rejected: porting `close_paraphrase` too.** It is portable — its own docstring
calls the client a port target — but it needs `lemma.v1.tsv`, `stopwords.v1.txt`
and the suffix rules, and a fence that renders green in the browser and red on
the server costs more trust than the latency it saves.

**Revisit if.** A check is added whose client and server answers can diverge. The
rule is that a live answer must be *provably* the gate's answer, pinned by a
test that fails on drift — not merely believed to be.

---

## Standing rule

A gate that can pass by not running is not a gate. Three in this repo could:
the parity gate skipped without node, the Postgres leg of the repository
contract was never parametrized, and the "no torch in the venv" check ran
`pip list` against a python that was not the venv. Each is now asserted to have
actually run.
