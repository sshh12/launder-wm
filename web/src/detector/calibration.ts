/**
 * sigma(T) -> z. TS port of `launder_core.detect.calibration`.
 *
 * `z = (score - 0.5) / sigma_null(n_scored)` (§4.2). The needle displays `z`,
 * not `score`: the weighted-mean score lives in ~[0.500, 0.555] and its
 * threshold moves with length, so a needle showing that is a needle showing
 * noise. `z` is standard deviations of watermark evidence above the human
 * baseline, with the notch at a CONSTANT `z*` (FPR 1%).
 *
 * ---------------------------------------------------------------------------
 * Why the curve is not just the closed form (§14.2 item 7)
 * ---------------------------------------------------------------------------
 * `sigmaClosedForm = 1/(2*sqrt(m*T))` is the null sd of an UNWEIGHTED mean of
 * `m*T` independent Bernoulli(1/2)s. Two things separate it from the truth, and
 * they are not the same kind of thing:
 *
 *  1. **Algebra.** The shipped detector takes a WEIGHTED mean
 *     (`w = linspace(10,1,m)`, renormalized so `sum(w) == m`), whose null sd is
 *     `sqrt(sum(w^2)/m)` times the above — `WEIGHTING_KAPPA` = 1.1128924 at
 *     m = 30, with no correlation involved. This is most of the measured kappa:
 *     all seven shipped buckets sit at 0.947-0.998 of it. **Editing the weight
 *     vector therefore invalidates every threshold and every packed
 *     `expected_z` for a purely algebraic reason.**
 *  2. **Measurement.** The residual on top of that — sampling noise plus
 *     whatever row-to-row structure real text has. Random token ids have no
 *     repeated n-grams and so cannot answer §14.2 item 7's actual question
 *     (how correlated are the rows in English); that remains open.
 *
 * So: the closed form gives the SHAPE, `WEIGHTING_KAPPA` gives the bulk of the
 * LEVEL analytically, and the measured percentile gives the rest. `sigmaNull` is
 * that product.
 *
 * ---------------------------------------------------------------------------
 * THE TWO THINGS THIS FILE HAD TO AGREE WITH PYTHON ON, AND NOW DOES
 * ---------------------------------------------------------------------------
 * 1. **The interpolation axis is `log n`, not `n`.** `forge calibrate` samples
 *    geometrically spaced buckets (40,60,80,120,180,260,400) and the shipped
 *    file declares `"kind": "piecewise_linear_in_log_n"`. Interpolating the
 *    same knots linearly in `n` moved `z` by 0.2% at a 157-row passage — small,
 *    invisible, and different on every keystroke between two buckets. Core
 *    interpolates in `log n`; so does this.
 * 2. **The bucket carries `sigma`, and this file will not derive it.** Core
 *    accepts a bucket with only `score_at_fpr` and converts it with
 *    `Phi^-1(1-fpr)`. Porting Acklam + a Halley step + `erfc` into TypeScript
 *    to reproduce that conversion to the last bit would be a second numerical
 *    implementation on a path nobody exercises — the definition of a silent
 *    divergence trap. `forge calibrate` now writes `sigma` explicitly (core
 *    prefers it when present, so both sides read ONE number), and
 *    `parseCalibration` refuses a bucket without it, naming the fix.
 */

import { tournamentWeights } from "./weightedmean.js";

/** One measured point of the null distribution. */
export interface CalibrationBucket {
  /** number of scored (unmasked) rows this bucket was measured at */
  readonly n_scored: number;
  /** the measured null sd of `score` at that length */
  readonly sigma: number;
  /** the `(1-fpr)` percentile it came from, kept for review; never read here */
  readonly score_at_fpr?: number;
  readonly n_samples?: number;
}

export interface Calibration {
  readonly schema: string;
  /** the notch, `Phi^-1(1-fpr)`. FPR 1% => 2.3263478740408. */
  readonly z_star: number;
  /** tournament depth `m` the closed form is computed at. */
  readonly depth: number;
  readonly fpr: number;
  /** ascending in `n_scored`; kappa piecewise-linear in `log n`, clamped. */
  readonly buckets: readonly CalibrationBucket[];
  /** `"closed_form"` or a description of where the numbers came from. */
  readonly source: string;
  /**
   * `kappa` when `buckets` is empty. NOT 1: `sigmaClosedForm` is the UNWEIGHTED
   * closed form and the shipped weighted mean's null sd is `WEIGHTING_KAPPA`
   * times it, analytically. Leaving it at 1 overstated every fallback `z` by
   * 5-11%, i.e. ran a 1%-FPR notch at a true 2-3%. Mirrors
   * `launder_core.detect.calibration.Calibration.base_kappa`.
   */
  readonly base_kappa?: number;
}

/**
 * `sqrt(sum(w^2)/m)` for `tournamentWeights(m)` — the exact factor by which the
 * weighted mean's null sd exceeds `sigmaClosedForm`, under independence.
 * Derived from the weights, not a literal, so editing them moves this with them.
 */
export function weightingKappa(depth: number): number {
  const w = tournamentWeights(depth);
  let sq = 0;
  for (const v of w) sq += v * v;
  return Math.sqrt(sq / depth);
}

/** The shipped value at m = 30: 1.1128924007211063. */
export const WEIGHTING_KAPPA = weightingKappa(30);

/**
 * `Phi^-1(0.99)` as the plan spells it. Only the FALLBACK: whenever a
 * calibration file is present its own `z_star` wins, exactly as in core.
 */
export const Z_STAR = 2.3263;

/** The closed-form null sd of the weighted mean: `1 / (2*sqrt(m*n))`. */
export function sigmaClosedForm(nScored: number, depth: number): number {
  if (nScored <= 0 || depth <= 0) return Number.POSITIVE_INFINITY;
  return 1 / (2 * Math.sqrt(depth * nScored));
}

/**
 * The shipped default: `data/assets/thresholds.v1.json`, inlined so a browser
 * that never fetches `calibration_url` still reads the same scale the server
 * does. Regenerate with `forge calibrate`; `web/tools/parity.mjs` asserts this
 * table is bucket-for-bucket the shipped file.
 *
 * PROVENANCE, STATED PLAINLY: 140,000 synthetic null sequences of uniform
 * random token ids, 20,000 per bucket. Random ids are not English — they have
 * no repeated n-grams, so the rows are independent and the measured kappa
 * collapses to the depth-weighting factor (~1.11) with none of the correlation
 * the plan warns about. §6.5 requires negatives that are human prose AND
 * unwatermarked Gemma-3 output under a different key. This is a first cut with
 * honest provenance, not a calibration.
 */
export const DEFAULT_CALIBRATION: Calibration = {
  schema: "launder.thresholds/1",
  z_star: 2.326347874040837,
  depth: 30,
  fpr: 0.01,
  buckets: [
    { n_scored: 36, sigma: 0.016569558645741677, n_samples: 20000 },
    { n_scored: 56, sigma: 0.01323311056491881, n_samples: 20000 },
    { n_scored: 76, sigma: 0.011508314210654739, n_samples: 20000 },
    { n_scored: 116, sigma: 0.009413290155116915, n_samples: 20000 },
    { n_scored: 176, sigma: 0.007386911346149124, n_samples: 20000 },
    { n_scored: 256, sigma: 0.006270549890797409, n_samples: 20000 },
    { n_scored: 396, sigma: 0.004833173686238122, n_samples: 20000 },
  ],
  source: "data/assets/thresholds.v1.json (synthetic-random-ids, 20000/bucket)",
};

/** The closed-form fallback: right shape, right weighting, unverified residual. */
export const CLOSED_FORM_CALIBRATION: Calibration = {
  schema: "launder.thresholds/1",
  z_star: Z_STAR,
  depth: 30,
  fpr: 0.01,
  buckets: [],
  base_kappa: WEIGHTING_KAPPA,
  source: "closed_form",
};

/**
 * `kappa(n) = sigma_measured / sigma_closed`, piecewise-linear in `log n`,
 * CLAMPED at both ends.
 *
 * Clamping rather than extrapolating is deliberate (and is core's docstring):
 * beyond the measured range the closed form's shape is the only thing we trust,
 * so keep the nearest measured level and let `sqrt(n)` do the rest.
 * Extrapolating a fitted slope down to a 10-row prefix would put a confident
 * nonsense number on the intro slider, where short prefixes are the point.
 */
export function kappa(nScored: number, cal: Calibration): number {
  const b = cal.buckets;
  if (b.length === 0) return cal.base_kappa ?? weightingKappa(cal.depth);
  const t = Math.max(nScored, 1);
  const bucketKappa = (k: CalibrationBucket): number => {
    const closed = sigmaClosedForm(k.n_scored, cal.depth);
    if (!Number.isFinite(closed) || closed <= 0) return 1;
    return k.sigma / closed;
  };
  const first = b[0] as CalibrationBucket;
  const last = b[b.length - 1] as CalibrationBucket;
  if (b.length === 1 || t <= first.n_scored) return bucketKappa(first);
  if (t >= last.n_scored) return bucketKappa(last);
  let lo = first;
  for (let i = 1; i < b.length; i++) {
    const hi = b[i] as CalibrationBucket;
    if (t <= hi.n_scored) {
      const span = Math.log(hi.n_scored) - Math.log(lo.n_scored);
      const frac = span === 0 ? 0 : (Math.log(t) - Math.log(lo.n_scored)) / span;
      const kl = bucketKappa(lo);
      return kl + frac * (bucketKappa(hi) - kl);
    }
    lo = hi;
  }
  return bucketKappa(last);
}

/** `sigma_null(n) = kappa(n) * 1/(2*sqrt(m*n))`. */
export function sigmaNull(nScored: number, cal: Calibration): number {
  const closed = sigmaClosedForm(nScored, cal.depth);
  if (!Number.isFinite(closed)) return Number.POSITIVE_INFINITY;
  return kappa(nScored, cal) * closed;
}

/**
 * `z = (score - 0.5) / sigma_null(n_scored)`. Zero scored rows => z = 0.
 *
 * A text with no scored positions carries no evidence either way, and `z = 0`
 * is exactly that statement — returning Infinity or NaN would propagate into
 * the needle, the gate and the leaderboard.
 */
export function zScore(score: number, nScored: number, cal: Calibration): number {
  const s = sigmaNull(nScored, cal);
  if (!Number.isFinite(s) || s <= 0) return 0;
  return (score - 0.5) / s;
}

/** `tau(eps) = 1/2 + z* * sigma(n)` — the score the notch sits at. */
export function thresholdScore(nScored: number, cal: Calibration): number {
  const s = sigmaNull(nScored, cal);
  if (!Number.isFinite(s)) return 0.5;
  return 0.5 + cal.z_star * s;
}

function bucketsFrom(entries: unknown, where: string): CalibrationBucket[] {
  if (!Array.isArray(entries)) throw new Error(`thresholds: ${where} is not an array`);
  const out = entries.map((e, i) => {
    const q = e as Record<string, unknown>;
    const n = q["n_scored"];
    const sigma = q["sigma"];
    if (typeof n !== "number") throw new Error(`thresholds: ${where}[${i}] has no n_scored`);
    if (typeof sigma !== "number" || !(sigma > 0)) {
      throw new Error(
        `thresholds: ${where}[${i}] (n_scored=${String(n)}) has no positive 'sigma'. ` +
          "The browser deliberately does not re-derive it from 'score_at_fpr' — that " +
          "would need a second Phi^-1 in TypeScript. Rebuild the file with a current " +
          "`forge calibrate`, which writes 'sigma' alongside 'score_at_fpr'.",
      );
    }
    const b: CalibrationBucket = { n_scored: n, sigma };
    const s = q["score_at_fpr"];
    const ns = q["n_samples"];
    return {
      ...b,
      ...(typeof s === "number" ? { score_at_fpr: s } : {}),
      ...(typeof ns === "number" ? { n_samples: ns } : {}),
    };
  });
  for (let i = 1; i < out.length; i++) {
    if ((out[i] as CalibrationBucket).n_scored <= (out[i - 1] as CalibrationBucket).n_scored) {
      throw new Error("thresholds: buckets must be strictly ascending in n_scored");
    }
  }
  return out;
}

/** A calibration group may repeat `z_star`/`fpr`, never contradict it. */
function assertNoShadow(
  group: Record<string, unknown>,
  key: string,
  top: number,
  name: string,
): void {
  const value = group[key];
  if (typeof value !== "number") return;
  if (value !== top) {
    throw new Error(
      `thresholds: calibrations.${name}.${key} = ${value} contradicts the top-level ` +
        `${key} = ${top}. The top-level value is THE knob (TECH_PLAN.md §12 row 8); a ` +
        `group key that overrode it made the documented edit inert. Delete ` +
        `calibrations.${name}.${key}, or change both.`,
    );
  }
}

/**
 * Parse a `thresholds.v1.json`, in the exact shape `forge calibrate` writes and
 * `launder_core.detect.calibration.parse_thresholds` reads. Rejects anything it
 * does not fully understand — a silently half-parsed calibration is a needle
 * reading the wrong scale.
 *
 * A flat top-level `buckets` list is accepted as `"default"`, matching core.
 */
export function parseCalibration(raw: unknown, name = "default"): Calibration {
  if (typeof raw !== "object" || raw === null) throw new Error("thresholds: not an object");
  const o = raw as Record<string, unknown>;
  if (o["schema"] !== "launder.thresholds/1") {
    throw new Error(`thresholds: unknown schema ${String(o["schema"])}`);
  }
  let fpr = typeof o["fpr"] === "number" ? o["fpr"] : 0.01;
  let zStar = typeof o["z_star"] === "number" ? o["z_star"] : Z_STAR;
  const depth = typeof o["depth"] === "number" ? o["depth"] : 30;

  let buckets: CalibrationBucket[];
  const groups = o["calibrations"];
  if (typeof groups === "object" && groups !== null) {
    const g = groups as Record<string, unknown>;
    if (!(name in g)) {
      throw new Error(
        `thresholds: calibration bucket set "${name}" not found; it has ` +
          `${JSON.stringify(Object.keys(g))}`,
      );
    }
    const group = g[name] as Record<string, unknown>;
    // THE TOP-LEVEL z_star AND fpr ARE THE KNOBS AND ARE NOT SHADOWABLE.
    // These two lines used to be `if (typeof group[k] === "number") x = group[k]`,
    // which silently overrode them — so §12's documented "edit z_star, redeploy"
    // did nothing and the real control was an undocumented nested key. A group
    // may repeat the value (the shipped file does); contradicting it is a load
    // failure, matching `launder_core.detect.calibration.parse_thresholds`.
    assertNoShadow(group, "fpr", fpr, name);
    assertNoShadow(group, "z_star", zStar, name);
    buckets = bucketsFrom(group["buckets"], `calibrations.${name}.buckets`);
  } else {
    if (name !== "default") {
      throw new Error(
        `thresholds: flat 'buckets' list carries only a "default" calibration, but ` +
          `"${name}" was requested`,
      );
    }
    buckets = bucketsFrom(o["buckets"], "buckets");
  }

  return {
    schema: "launder.thresholds/1",
    z_star: zStar,
    depth,
    fpr,
    buckets,
    // Only consulted if the file carried no buckets, i.e. if it degenerated to
    // the fallback anyway.
    base_kappa: weightingKappa(depth),
    source: typeof o["source"] === "string" ? o["source"] : "parsed",
  };
}
