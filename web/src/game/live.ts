/**
 * The live half of the gate — TECH_PLAN.md §7.1, §8.3, §10.5.
 *
 * Seven of this game's checks are closed-form functions of text the browser is
 * already holding. Until now the browser computed exactly one of them live (the
 * detector threshold, as the needle) and made the player press Check to learn
 * the other six — so "Real characters" and "At least 50 words", two rules that
 * can be answered in under a millisecond, were reported by a network round trip
 * that also spends money on a language model.
 *
 * That is the wrong shape for a puzzle game. A constraint the player can watch
 * is a mechanic; a constraint discovered only on rejection is a bug report
 * (`close_paraphrase.py` says exactly this about its own metrics). It is also
 * how you end up with a level whose "Verbatim phrase" rule never names the
 * phrase: nothing on screen moves when you break it.
 *
 * **THE SERVER IS STILL AUTHORITY** (§8.3). Nothing here decides anything. The
 * moment a `trace` arrives, `Gate.statusFor` prefers it over every value this
 * module produced, and a submission is scored, ranked and persisted from the
 * server's recomputation alone. This is the same standing the "N changed"
 * counter has always had, and it is what makes publishing the watermark keys
 * costless to integrity: a client that lies to itself is lying to itself.
 *
 * **WHAT IS DELIBERATELY ABSENT.** Three checks are not here and must not be
 * added without a port that carries its own parity gate:
 *
 * - `llm_gate` — a paid model reading the text. There is no client-side answer.
 * - `unit_test` — runs a suite server-side.
 * - `close_paraphrase` — portable in principle (its own docstring calls this
 *   file a port target) but it needs `lemma.v1.tsv`, `stopwords.v1.txt` and the
 *   suffix rules, and a fence that renders green in the browser and red on the
 *   server is worse than one that stays quiet until Check.
 *
 * A check with no entry in `EVALUATORS` returns `null` and keeps the honest
 * "Not checked yet" pip, which is the whole reason that string exists.
 */

import { formatPoints, type PointsScale } from "./needle";
import { score } from "../scoring";
import { normalize, words } from "../scoring/normalize";
import { classify, offendingCount, type HomoglyphPolicy } from "../scoring/unicode";
import type { CheckSpecWire, Reading } from "../state";
import type { DistanceResult } from "../scoring/damerau";

/** A live verdict. `params` uses the same keys the server's `CheckResult.params`
 *  uses, so a copy template renders identically from either source. */
export interface LiveResult {
  status: "pass" | "fail";
  params: Record<string, string | number>;
}

/** Everything an evaluator may read. Nothing else is in scope — in particular
 *  there is no access to the DOM, so this module is testable as pure logic. */
interface LiveContext {
  /** the pristine passage, as the server inlined it */
  readonly original: string;
  /** the player's current text, RAW — normalization is each check's own
   *  business, exactly as it is server-side */
  readonly raw: string;
  /** `normalize(raw)`, computed once per text */
  readonly normalized: string;
  /** `words(raw)`, computed once per text */
  readonly words: readonly string[];
  /** `score(original, raw)`, computed once per text and shared by the two
   *  checks that need a backtrace. ONE distance function in the product. */
  readonly diff: DistanceResult;
  /** the passage's `rules.locked_phrases`, from the boot payload */
  readonly lockedPhrases: readonly string[];
  /** the reading the needle is showing, or null before the first one lands */
  readonly reading: Reading | null;
  /** the z -> 0..100 relabelling, so a live rejection preview and the meter
   *  above it cannot print two different number systems */
  readonly points: PointsScale;
}

type Evaluator = (
  params: Record<string, unknown>,
  ctx: LiveContext,
) => LiveResult | null;

const num = (v: unknown, fallback: number): number => {
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : fallback;
};

const EVALUATORS: Record<string, Evaluator> = {
  /* --- SANITATION ---------------------------------------------------- */
  unicode_sanitation: (params, ctx) => {
    const declared = params.reject_categories;
    const categories = new Set<string>(
      Array.isArray(declared) ? declared.map((c) => String(c)) : [],
    );
    // `homoglyph` and `combining_marks` are governed by their own params, so
    // they are always live even though levels.toml does not list them among the
    // categories. Same two lines as the Python check.
    categories.add("homoglyph");
    categories.add("combining_marks");
    const counts = classify(ctx.normalized, {
      rejectCategories: categories,
      homoglyphPolicy: String(params.homoglyph_policy ?? "allow") as HomoglyphPolicy,
      maxCombiningMarks: num(params.max_combining_marks, 2),
      allowedChars: new Set(normalize(ctx.original)),
    });
    const count = offendingCount(counts);
    return { status: count === 0 ? "pass" : "fail", params: { count } };
  },

  /* --- LENGTH -------------------------------------------------------- */
  word_floor: (params, ctx) => {
    const minWords = num(params.min_words, 0);
    const nWords = ctx.words.length;
    return {
      status: nWords < minWords ? "fail" : "pass",
      params: { n_words: nWords, min_words: minWords },
    };
  },

  /* --- SHAPE --------------------------------------------------------- */
  edit_budget: (params, ctx) => {
    const budget = num(params.max_word_distance, Number.POSITIVE_INFINITY);
    const distance = ctx.diff.distance;
    return {
      status: distance > budget ? "fail" : "pass",
      params: { distance, max_word_distance: budget },
    };
  },

  locked_phrase: (params, ctx) => {
    // A level that runs this check against a passage declaring no phrases is a
    // GateDataError server-side, not a pass. Staying silent here rather than
    // rendering a green pip keeps the client from contradicting that.
    if (ctx.lockedPhrases.length === 0) return null;
    const exact = String(params.match ?? "whitespace_insensitive") === "exact";
    const haystack = exact ? ctx.raw : ctx.normalized;
    for (const phrase of ctx.lockedPhrases) {
      const needle = exact ? phrase : normalize(phrase);
      if (!haystack.includes(needle)) return { status: "fail", params: { phrase } };
    }
    return { status: "pass", params: { phrase: ctx.lockedPhrases[0] as string } };
  },

  edit_region: (params, ctx) => {
    const window = num(params.editable_prefix_words, 0);
    if (window < 1) return null; // a config error, which the server reports
    // `i` indexes the ORIGINAL word sequence, which is the sequence the window
    // is defined over — the player is told "the first N words of the passage",
    // and the passage is what they can see.
    let outside = 0;
    let first = 0;
    for (const op of ctx.diff.ops) {
      if (op.i < window) continue;
      outside += 1;
      if (first === 0 || op.i + 1 < first) first = op.i + 1;
    }
    return {
      status: outside > 0 ? "fail" : "pass",
      params: {
        editable_prefix_words: window,
        n_outside: outside,
        first_bad_word_index: first,
      },
    };
  },

  /* --- DETECTOR ------------------------------------------------------ *
   * These two are the needle's own state under a different glyph, not a
   * verdict of our own — which is why they were the only live rows before
   * this module existed. Both bounds are RELATIVE TO THE NOTCH, so a
   * recalibration moves them together and neither is hardcoded here.
   * ------------------------------------------------------------------- */
  detector_threshold: (params, ctx) => {
    const reading = ctx.reading;
    if (reading === null) return null;
    const maxZ = num(params.max_z, 0);
    return {
      status: reading.z - reading.zStar > maxZ ? "fail" : "pass",
      params: {
        z_display: formatPoints(reading.z, ctx.points),
        z_star_display: formatPoints(reading.zStar, ctx.points),
        z_target_display: formatPoints(reading.zStar + maxZ, ctx.points),
      },
    };
  },

  detector_floor: (params, ctx) => {
    const reading = ctx.reading;
    if (reading === null) return null;
    const minZ = num(params.min_z, Number.NEGATIVE_INFINITY);
    return {
      status: reading.z - reading.zStar < minZ ? "fail" : "pass",
      params: {
        z_display: formatPoints(reading.z, ctx.points),
        z_star_display: formatPoints(reading.zStar, ctx.points),
        z_floor_display: formatPoints(reading.zStar + minZ, ctx.points),
      },
    };
  },
};

/** The check names this module can answer. Exported so a test can assert it
 *  against the level's own check list rather than against a second copy. */
export const LIVE_CHECKS: readonly string[] = Object.keys(EVALUATORS);

export interface LiveOptions {
  readonly original: string;
  readonly lockedPhrases: readonly string[];
  readonly points: PointsScale;
}

/**
 * Holds the derived-once-per-text values and answers one check at a time.
 *
 * Everything derived from the text is computed lazily and cached against the
 * exact string it came from: `Gate.rebuild()` runs on every keystroke and once
 * per check per rebuild, so a naive implementation would re-run the O(m·n)
 * Damerau table half a dozen times per character typed.
 */
export class LivePreview {
  private raw: string;
  private reading: Reading | null = null;
  private cachedFor: string | null = null;
  private cache: { normalized: string; words: readonly string[]; diff: DistanceResult } | null =
    null;

  constructor(private readonly o: LiveOptions) {
    this.raw = o.original;
  }

  setText(text: string): void {
    this.raw = text;
  }

  setReading(reading: Reading | null): void {
    this.reading = reading;
  }

  /** The live word distance, off the SAME backtrace the budget rule reads.
   *  The counter under the textarea and the budget pip are one number computed
   *  once, so they cannot disagree — the property `edit_budget.py` was written
   *  to guarantee, now held on this side of the wire too. */
  distance(): number {
    return this.derived().diff.distance;
  }

  /** A live verdict for one check, or `null` when this check has no honest
   *  client-side answer. */
  evaluate(spec: CheckSpecWire): LiveResult | null {
    const evaluator = EVALUATORS[spec.check];
    if (evaluator === undefined) return null;
    const derived = this.derived();
    return evaluator(spec.params ?? {}, {
      original: this.o.original,
      raw: this.raw,
      normalized: derived.normalized,
      words: derived.words,
      diff: derived.diff,
      lockedPhrases: this.o.lockedPhrases,
      reading: this.reading,
      points: this.o.points,
    });
  }

  private derived(): { normalized: string; words: readonly string[]; diff: DistanceResult } {
    if (this.cachedFor === this.raw && this.cache !== null) return this.cache;
    const normalized = normalize(this.raw);
    this.cache = {
      normalized,
      words: words(this.raw),
      diff: score(this.o.original, this.raw),
    };
    this.cachedFor = this.raw;
    return this.cache;
  }
}
