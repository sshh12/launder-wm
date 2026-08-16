/**
 * The int64 LCG hash. TS port of `launder_core.watermark.hash`.
 *
 * TECH_PLAN.md §4.2:
 *
 *     h = 1                                   # IV, int64
 *     for tok in ngram:
 *         h = wrap64(wrap64(h + tok) * 6364136223846793005 + 1)
 *
 * ---------------------------------------------------------------------------
 * DIVERGENCE TRAP #2 (§4.3): int64 wraparound.
 * ---------------------------------------------------------------------------
 * Every arithmetic step upstream is `torch.int64`, which wraps mod 2**64 in
 * two's complement. `BigInt` does not wrap — it grows without bound — so the
 * *whole* point of this module is that `wrap64` is applied after EVERY add and
 * EVERY multiply, never once at the end. Wrapping only at the end is not the
 * same function: `wrap64(wrap64(a) * b) != wrap64(a * b)` in general, because
 * the discarded high bits of `a` re-enter the product's low 64 bits through
 * the multiply.
 *
 * The reference is verified against numpy int64 in
 * `tools/reference/ref.py` (2,000 random cases, exact match), and the depth
 * keys below are pinned as golden case #1.
 */

/** The LCG multiplier. Same constant in torch, numpy, Python and here. */
export const LCG_MULTIPLIER = 6364136223846793005n;

/** The LCG increment. */
export const LCG_INCREMENT = 1n;

/**
 * The hash initialisation vector.
 *
 * `1n` is the HF-transformers / `synthid-text==0.2.1` value and therefore ours
 * (§4.1). The `google-deepmind/synthid-text@main` variant instead uses
 * `int.from_bytes(sha256(keys.tobytes()),'big') % (2**63-1)` and is
 * bit-incompatible with everything HF generates. Do not "upgrade" this.
 */
export const HASH_IV = 1n;

/**
 * Two's-complement 64-bit wrap. This is the whole of trap #2.
 *
 * `BigInt.asIntN(64, x)` keeps the low 64 bits and reinterprets bit 63 as the
 * sign, which is exactly what `torch.int64` does on overflow.
 */
export function wrap64(x: bigint): bigint {
  return BigInt.asIntN(64, x);
}

/**
 * Fold one token into the running hash.
 *
 * Three wraps, in the same order as the Python:
 *   `wrap(wrap(wrap(h + tok) * MULT) + 1)`
 */
export function accumulateStep(h: bigint, tok: bigint): bigint {
  return wrap64(wrap64(wrap64(h + tok) * LCG_MULTIPLIER) + LCG_INCREMENT);
}

/**
 * Fold a whole sequence into `cur`.
 *
 * @param cur  the running hash (`HASH_IV` to start a fresh n-gram)
 * @param data token ids, folded left to right
 */
export function accumulateHash(cur: bigint, data: ArrayLike<number>): bigint {
  let h = cur;
  for (let i = 0; i < data.length; i++) {
    h = accumulateStep(h, BigInt(data[i] as number));
  }
  return h;
}

/**
 * The per-depth watermark keys for one n-gram.
 *
 * Note the shape, which is the single easiest thing to get subtly wrong: the
 * FULL n-gram is hashed first, and then each watermark key is folded in as one
 * *additional* accumulate step on top of that shared prefix hash. It is not
 * `accumulate(1, [key, ...ngram])` and it is not `accumulate(key, ngram)`.
 *
 * Pinned by golden case #1: ngram `[1,235280,2121,576,573]` with the first five
 * canonical keys gives
 * `[-6504205589568445072, 318672039786673866, 8070454580555681646,
 *   8340369110341966617, 5852124156879677502]`.
 */
export function ngramDepthKeys(
  ngram: ArrayLike<number>,
  keys: readonly number[],
): BigInt64Array {
  const shared = accumulateHash(HASH_IV, ngram);
  const out = new BigInt64Array(keys.length);
  for (let d = 0; d < keys.length; d++) {
    out[d] = accumulateStep(shared, BigInt(keys[d] as number));
  }
  return out;
}

/**
 * The context hash of one row: `accumulate(1, t[i .. i+n-2])`, i.e. the `n-1`
 * LEADING tokens of the window, excluding the current token.
 *
 * Pinned by golden case #1: `acc(1,[1,235280,2121,576]) = -4495156567014905553`.
 */
export function contextHash(window: ArrayLike<number>): bigint {
  return accumulateHash(HASH_IV, window);
}
