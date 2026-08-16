# Launder WM — spec

**Pitch:** A craft game. You're given real watermarked text and you launder the watermark out — dropping the detector below its threshold in as few edits as you can. Low floor (anyone can rewrite their way out), high ceiling (the surgical few-word kill takes the top of that level's board).

**What it rides:** the live discourse around AI text watermarking (Anthropic's SynthID-based watermark, Aug 2026). You can't self-host *Claude's* watermark — its key is secret and server-side. So the game runs the full watermark→detect loop on an **open model** (Gemma/GPT-2 via `google-deepmind/synthid-text`). The detector is therefore *real*, not a fake "AI %". For the HN crowd that's the selling point: real watermark, real detector, self-hosted, you hold the key.

## Core loop

A **raw textarea pre-filled with the watermarked passage** (never a blank box), one watermark **needle**, one **Check** button. Edit the text → watch the needle fall → submit when it's under the line. Cleared = needle past the threshold notch, chime.

## Two systems (this is the heart of it)

- **Watermark detector — local, fast, free → runs live.** Debounced on every keystroke. Drives the needle *and* a passive **heat-mirror** under the box that re-colors each token by how much watermark signal it carries. Because token scores are keyed to the *preceding* context, an edit also cools the words right after it — a visible **ripple** rightward. This ripple is the main juice and teaches the deep skill wordlessly: cut *upstream* of a hot run, not downstream of it.
- **LLM judge — cheap model, fires only on Submit.** Not a score you optimize — a **gate**, asked one question: *"Is this natural, human-readable prose that preserves the original's meaning? yes/no + why."* Temp 0, cached per (passage, submission) hash, so cost is ~one call per serious attempt. Its rejection reason is shown as feedback ("meaning drifted — you dropped the claim about X"). The gate is thematically the point: you launder text past one AI while being judged natural by another.

The gate closes every degeneracy a textarea opens, in one shot: garbage ("the the the") fails *natural*; zero-width/soft-hyphen tokenizer exploits fail *natural*; keyword-preserving nonsense fails *meaning*. It deliberately **does not** close whole-sentence paraphrase — that's legitimate laundering (swapping high-optionality prose for plain prose genuinely lowers the watermark), just a high-edit-distance clear. That's the valid-but-unimpressive floor.

## Scoring & par

- **Score = word-level Damerau–Levenshtein distance** (changed or moved words; a reorder counts as 1, so reordering is *cheap, not free*), computed on submit. Lower is better. Displayed as "N words changed."
- **Par:** the move space is unbounded, so minimum edit distance can't be computed. Set par from an authored reference solution per level, or from the best clear in the first N plays (self-balancing).

## The skill, and the one thing not to get wrong

Detection strength scales with **text length** and **token entropy** — the watermark can only bite where the model had many near-equal choices (high optionality). The honest skill is: read for *optionality*, attack upstream of hot runs, minimize edits.

**Critical guardrail:** the watermark is **not** about style. It boosts whatever the secret key says, not "fancy" or "AI-sounding" words. The design must never reward a "spot the pretentious word" heuristic — that heuristic is false, and teaching it defeats the point. Route the skill through entropy/position, never vibe.

*(Both the length/entropy relationship and the exact context-window size behind the ripple should be verified against the reference detector before final tuning.)*

## Intro (wordless tutorial)

One sentence, the needle, and a **length slider**. Drag right → more of a pre-written watermarked passage is revealed → the needle climbs from "could be human" into "detected." Teaches *longer = more evidence* with zero copy. Then a **≥50-word floor** snaps in ("you can't just delete your essay"), killing the obvious "make it shorter" move and motivating rewriting. Hand off to level 1.

## Rulesets (the gate is a checklist; each ruleset adds one check)

These are the **rulesets**, not the campaign positions — one word, one meaning. A campaign level is a passage plus one of these; `data/config/progression.toml` says which. The player sees the ruleset's name ("Clean it"), never the id.

| id | Ruleset | Adds to gate | Teaches |
|---|-------|-------------|---------|
| L1 | Clean it | natural + meaning | the loop, the ripple |
| L2 | Budget | ≤ N words changed | placement — cut upstream of a hot run |
| L3 | Locked | a phrase must appear verbatim | launder *around* fixed signal ("quote your source") |
| L4 | Tight fence | meaning clamps to close-paraphrase-only | surgical, faithful edits; cheap rewrites stop working |
| L5 | Code | unit test must pass | why code is a terrible host (almost no entropy to hide in) |
| L6 | Regenerate locally | — | long high-entropy wall; a "regenerate locally" option clears it instantly → the one robust attack: **control the sampler** |

L5's `unit_test` check needs a dependency nobody has wired, so the ruleset is dropped at boot and is not in the campaign; L6 has no authored passage. The shipped campaign runs L1 → L4.

## The campaign & share

A **linear 15-level campaign**, each level one passage with a shown par, played in order: clear level `n` and level `n+1` unlocks. Difficulty is the ordering — L1 through L4, and inside each group the detector's expected reading climbs. There is no date, no rollover and nothing to come back tomorrow for: a visitor who arrived from a link can play the whole thing now. Progress is anonymous: kept in `localStorage` so the page knows which level to open before it paints, and mirrored on the server against the same anonymous session id so the two can be reconciled at boot. Neither is an account. `?level=<n>` jumps straight to a level.

Share headlines **words changed** ("Launder WM — level 3 cleared in 4 🧼"), and clearing the last level produces one line per five levels with every score on it; the brag is the **diff itself** — the 4-word change that killed a 90% detection.

## UX principles (neal.fun-simple)

One screen throughout. No accounts (localStorage progress). Instant play. Pre-filled box — no blank-page freeze. Live feedback is free (local detector); the only paid call is one cheap gate per submit.

## Scope

**v1:** raw textarea; live needle + heat-mirror + ripple; gate (natural + meaning) on submit; edit-distance scoring; the 15-level campaign + par; the rulesets; slider intro.

**v2:** spoofing mode (make *human* text read as watermarked); versus mode (Smuggler hides a watermarked phrase, Narc must locate it); an over-scrubbing second threshold ("you tried too hard — now you look guilty").

## Risks

- It's a **craft** game, so it lives or dies on (a) tight, legible minimal-edit scoring and (b) being able to *see and compare* solutions (the diff). Both are core above; don't let them slip.
- **Passage authoring is the real work:** each level must be solvable-but-not-trivial with a genuine par, and the fifteen of them have to rise in difficulty in the order they are played. Budget for it; underestimating this sinks the game.
- **Latency:** the local detector must be fast, or animate the needle optimistically and reconcile on commit.