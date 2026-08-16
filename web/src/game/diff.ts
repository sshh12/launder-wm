/**
 * The diff exhibit — TECH_PLAN.md §10.1, §8.3.
 *
 * Renders `<ins>`/`<del>` markup from the SERVER's op list. It does not compute
 * distance and it does not compute an alignment of its own: `ops` comes from
 * launder_core.scoring.damerau, which is the authority, and two clients must
 * render the same ops identically (golden case #8).
 *
 * Diffs are public by design — the brag, the anti-cheat mechanism and the
 * teaching tool are the same object. Colour is never the only channel: <ins> is
 * underlined and <del> is struck (§10.4).
 *
 * Player text reaches the DOM through textContent only. Never innerHTML here:
 * the words being rendered are attacker-controlled by construction.
 */

import type { EditOpWire } from "../state";

export type DiffKind = "eq" | "ins" | "del";

export interface DiffSegment {
  kind: DiffKind;
  words: string[];
}

/** Whitespace word split, matching the unit the ops are indexed in. */
export function words(text: string): string[] {
  const trimmed = text.trim();
  return trimmed === "" ? [] : trimmed.split(/\s+/);
}

/**
 * Walk the two word sequences under the server's ops and emit a flat segment
 * list. Unknown or inconsistent ops degrade to "equal", never to a crash: a
 * malformed diff must not be able to blank the result card.
 */
export function diffSegments(
  original: readonly string[],
  submitted: readonly string[],
  ops: readonly EditOpWire[],
): DiffSegment[] {
  const del = new Map<number, EditOpWire>();
  const sub = new Map<number, EditOpWire>();
  const ins = new Map<number, EditOpWire>();
  const trans = new Map<number, EditOpWire>();
  for (const op of ops) {
    if (op.op === "del") del.set(op.i, op);
    else if (op.op === "sub") sub.set(op.i, op);
    else if (op.op === "ins") ins.set(op.j, op);
    else if (op.op === "transpose") trans.set(op.i, op);
  }

  const out: DiffSegment[] = [];
  const push = (kind: DiffKind, word: string): void => {
    const last = out[out.length - 1];
    if (last && last.kind === kind) last.words.push(word);
    else out.push({ kind, words: [word] });
  };

  let i = 0;
  let j = 0;
  const n = original.length;
  const m = submitted.length;
  const orig = (k: number): string => original[k] ?? "";
  const subm = (k: number): string => submitted[k] ?? "";
  let guard = 0;
  while ((i < n || j < m) && guard++ < n + m + 8) {
    if (j < m && ins.has(j)) {
      push("ins", subm(j));
      j += 1;
    } else if (i < n && del.has(i)) {
      push("del", orig(i));
      i += 1;
    } else if (i < n && sub.has(i)) {
      push("del", orig(i));
      if (j < m) push("ins", subm(j));
      i += 1;
      j += 1;
    } else if (i < n && trans.has(i)) {
      // One transposition is one edit but two words move.
      push("del", orig(i));
      if (i + 1 < n) push("del", orig(i + 1));
      if (j < m) push("ins", subm(j));
      if (j + 1 < m) push("ins", subm(j + 1));
      i += 2;
      j += 2;
    } else if (i < n && j < m) {
      push("eq", subm(j));
      i += 1;
      j += 1;
    } else if (j < m) {
      push("ins", subm(j));
      j += 1;
    } else {
      push("del", orig(i));
      i += 1;
    }
  }
  return out;
}

export function renderDiff(
  container: HTMLElement,
  original: readonly string[],
  submitted: readonly string[],
  ops: readonly EditOpWire[],
): void {
  const doc = container.ownerDocument;
  while (container.firstChild) container.removeChild(container.firstChild);
  for (const seg of diffSegments(original, submitted, ops)) {
    const el =
      seg.kind === "eq"
        ? doc.createElement("span")
        : doc.createElement(seg.kind === "del" ? "del" : "ins");
    el.textContent = `${seg.words.join(" ")} `;
    container.appendChild(el);
  }
}
