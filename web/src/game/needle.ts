/**
 * The needle — TECH_PLAN.md §10.4 ballistics and §10.6 "Dead Air".
 *
 * It reads values and paints an instrument. It never reads text and never
 * computes a score (§10.1). Every number it shows is a real detector output.
 *
 * Ballistics, coupled to the ripple: movement starts at 0ms and settles at
 * ~380ms, slightly AFTER the ripple's tail — that lag is what makes the two
 * read as one physical event.
 *   |dz| >  8% of scale -> 380ms cubic-bezier(.34,1.36,.64,1)  (overshoot)
 *   |dz| <= 8% of scale -> 160ms cubic-bezier(.4,0,.2,1)       (critically
 *                                                               damped; the
 *   damping on small moves is what makes the big swings feel earned)
 *
 * Threshold crossing is ONE moment: the notch fills, one 220ms hairline sweep,
 * one short chime, once — with 2%-of-scale hysteresis before it can re-arm, or
 * a value bouncing across the line re-fires it on every keystroke.
 *
 * THE WINDOW (`detector_floor`). Two levels give the scale a bottom as well as
 * a top: a reading far below the null is its own evidence, so the player has to
 * LAND between the floor and the notch rather than scrub as far left as they
 * can. Those levels get one extra printed mark and the shaded window it opens.
 * The floor is drawn in a different vocabulary from the notch on purpose — they
 * bound the same window but they do not mean the same thing, and two identical
 * ticks would read as a pair.
 */

import { readBoot, type CheckSpecWire, type Copy } from "../state";

export interface NeedleElements {
  meter: HTMLElement;
  face: HTMLElement;
  needle: HTMLElement;
  fill: HTMLElement;
  notch: HTMLElement;
  peak: HTMLElement;
  sweep: HTMLElement;
  tri: HTMLElement;
  floorLabel: HTMLElement;
  scaleRow: HTMLElement;
  num: HTMLElement;
  stateWord: HTMLElement;
  live: HTMLElement;
}

export interface NeedleConfig {
  zStar: number;
  scale: { min: number; max: number };
  copy: Copy;
  /**
   * The bottom of the window, IN z, or null on the levels that have no floor.
   *
   * Optional because the needle can find it for itself: it is a function of the
   * level's own `detector_floor` params and `z_star`, both of which the boot
   * payload already inlined into the page this instrument is painted on. A
   * caller that has the payload in hand is welcome to pass it and skip the
   * lookup — the two paths cannot disagree, because they read the same field.
   */
  floorZ?: number | null;
}

export const POINTS_MIN = 0;
export const POINTS_MAX = 100;

/** Everything `points()` needs; `NeedleConfig` satisfies it structurally, so
 *  the needle, the gate and the rail all relabel against the same numbers. */
export interface PointsScale {
  zStar: number;
  scale: { min: number; max: number };
}

/** Where z sits on the printed scale, as 0..100. This IS the scale — the
 *  points readout is a monotone relabelling of z onto the ruler already drawn
 *  under the needle, not a new quantity and emphatically not a probability. */
function raw(z: number, cfg: PointsScale): number {
  const span = cfg.scale.max - cfg.scale.min || 1;
  return ((z - cfg.scale.min) / span) * 100;
}

/**
 * The displayed number, 0..100.
 *
 * The instrument keeps working in z everywhere it matters — geometry,
 * thresholds, the wire, the DB — because z is the statistic. Only what the
 * player READS changes: z is a number nobody can place, and a scale printed
 * -2..10 reads as broken.
 *
 * The two corrections below are not cosmetic. Rounding can put a value that is
 * ABOVE the line onto the same integer as the line itself, and the readout must
 * never disagree with the verdict — that is exactly the "2.3 on both sides of
 * the line" bug the two-decimal z display was introduced to avoid, arriving
 * again through a coarser scale. So the SIDE OF THE LINE WINS OVER THE
 * ROUNDING: a z above z* is forced to at least pStar + 1, and a z at or below
 * z* is forced down to pStar.
 */
export function points(z: number, cfg: PointsScale): number {
  const pStar = Math.round(raw(cfg.zStar, cfg));
  let p = Math.max(POINTS_MIN, Math.min(POINTS_MAX, Math.round(raw(z, cfg))));
  if (z > cfg.zStar && p <= pStar) p = Math.min(pStar + 1, POINTS_MAX);
  if (z <= cfg.zStar && p > pStar) p = pStar;
  return p;
}

/** The wire name of the rule that gives the scale a bottom. A check name, not a
 *  player-facing string: it is the same token levels.toml and the gate's trace
 *  use, and the words the player reads for it live in copy.toml under
 *  `check.detector_floor` (§10.7). */
const FLOOR_CHECK = "detector_floor";

/**
 * The floor in z, or null when this level does not run the check.
 *
 * `min_z` is RELATIVE TO THE NOTCH, exactly like `detector_threshold`'s
 * `max_z`: the shipped -1.2 means "no further below z* than 1.2", not "no lower
 * than -1.2". Reading it as an absolute would put the mark at 8 on a face whose
 * notch is 36 — a floor above the line, which is not a window at all.
 */
export function floorZFromChecks(
  checks: readonly CheckSpecWire[] | undefined,
  zStar: number,
): number | null {
  const spec = checks?.find((c) => c.check === FLOOR_CHECK);
  const min = spec?.params?.["min_z"];
  // A check present but carrying no usable `min_z` draws NO mark. An invented
  // default would be a line the player can fail against that the server never
  // agreed to.
  if (typeof min !== "number" || !Number.isFinite(min)) return null;
  return zStar + min;
}

/** This level's checks, from the payload the server inlined. Absent or
 *  malformed is "no floor", never a thrown error: the instrument's job is to
 *  keep reading. */
function checksInPage(doc: Document): CheckSpecWire[] {
  try {
    return readBoot(doc).level.checks ?? [];
  } catch {
    return [];
  }
}

/**
 * The floor this instrument will draw AND speak, or null — ONE decision, made
 * once.
 *
 * A floor that does not open a window on the printed scale is not a floor, and
 * the guard belongs here rather than in the drawing: deciding it separately for
 * the mark and for the spoken value is exactly how a face that shows no bottom
 * ends up announcing one. Nothing in the shipped levels is degenerate; a params
 * typo that put `min_z` the right way up would be.
 */
function usableFloor(cfg: NeedleConfig, doc: Document): number | null {
  const z = cfg.floorZ ?? floorZFromChecks(checksInPage(doc), cfg.zStar);
  if (z === null) return null;
  return points(z, cfg) < points(cfg.zStar, cfg) ? z : null;
}

/** Never rendered with a `%` and never called a percentage: [readout]'s
 *  standing rule is that the product shows no "% AI" figure, and a bare 0-100
 *  number wearing a percent sign would read as exactly that. */
export function formatPoints(z: number, cfg: PointsScale): string {
  return String(points(z, cfg));
}

export class Needle {
  private readonly els: NeedleElements;
  private readonly cfg: NeedleConfig;
  private readonly reduceMotion: MediaQueryList | null;
  private faceWidth = 360;
  private dpr = 1;
  private value: number;
  private last: number;
  private peak: number;
  private armed = true;
  private cleared = false;
  private peakTimer: ReturnType<typeof setTimeout> | null = null;
  private frame: number | null = null;
  private queued: number | null = null;
  private audio: AudioContext | null = null;
  private soundOn = true;
  /** the bottom of the window in z, or null on a level with no floor */
  private readonly floorZ: number | null;
  onCross: ((below: boolean) => void) | null = null;

  constructor(els: NeedleElements, cfg: NeedleConfig, initialZ: number) {
    this.els = els;
    this.cfg = cfg;
    this.floorZ = usableFloor(cfg, els.meter.ownerDocument);
    this.value = initialZ;
    this.last = initialZ;
    this.peak = initialZ;
    const view = els.meter.ownerDocument.defaultView;
    this.reduceMotion =
      typeof view?.matchMedia === "function"
        ? view.matchMedia("(prefers-reduced-motion: reduce)")
        : null;
    // The meter speaks in points, so its bounds are the points bounds. They are
    // set here rather than in index.html's checked-in markup alone, because the
    // needle is the one place that knows what the number means.
    els.meter.setAttribute("aria-valuemin", String(POINTS_MIN));
    els.meter.setAttribute("aria-valuemax", String(POINTS_MAX));
    this.buildScale();
    this.buildWindow();
    this.measure();
    // The server positions the needle with `--init-x` (a percentage of the
    // face) so the instrument is correct before this file runs. From here the
    // position is a px transform; zero the CSS offset so the two never compose.
    this.els.needle.style.left = "0px";
    this.paint(initialZ, true);
    this.writeReadout(initialZ);
    if (typeof view?.ResizeObserver === "function") {
      new view.ResizeObserver(() => {
        this.measure();
        this.paint(this.value, true);
      }).observe(els.face);
    } else {
      view?.addEventListener("resize", () => {
        this.measure();
        this.paint(this.value, true);
      });
    }
  }

  /** 0..100 along the printed scale. */
  private pct(z: number): number {
    const { min, max } = this.cfg.scale;
    const span = max - min || 1;
    return Math.max(0, Math.min(100, ((z - min) / span) * 100));
  }

  private buildScale(): void {
    const doc = this.els.meter.ownerDocument;
    // 21 hairline ticks at 5%; majors at 25%. The ticks ARE the scale, which is
    // why --rule is held to 3:1 (§10.6).
    for (let p = 0; p <= 100; p += 5) {
      const t = doc.createElement("div");
      t.className = "tick";
      t.setAttribute("data-maj", p % 25 === 0 ? "1" : "0");
      t.style.left = `${p}%`;
      this.els.face.appendChild(t);
    }
    for (const p of [0, 25, 50, 75, 100]) {
      const s = doc.createElement("span");
      // The printed scale reads 0..100 because points ARE the position on the
      // face: the label at 25% of the way along is 25, by construction. It used
      // to print z, which is how a player ended up reading "-2" off the left end
      // of a meter and concluding the instrument was broken.
      s.textContent = String(p);
      if (p === 0) s.style.left = "0";
      else if (p === 100) {
        s.style.left = "100%";
        s.style.transform = "translateX(-100%)";
      } else {
        s.style.left = `${p}%`;
        s.style.transform = "translateX(-50%)";
      }
      this.els.scaleRow.appendChild(s);
    }
    const at = `${this.markPct(this.cfg.zStar)}%`;
    this.els.notch.style.left = at;
    this.els.tri.style.left = at;
    this.els.floorLabel.style.left = at;
    this.els.floorLabel.style.transform = "translateX(-50%)";
  }

  /**
   * Where a PRINTED LIMIT sits, as a percentage of the face.
   *
   * `points()`, not `pct()`, and the difference is the whole reason the two
   * functions are not one. `pct()` is the continuous position of a reading; the
   * needle uses it because a needle is a reading. A printed limit is a NUMBER
   * the instrument also prints in words — "Under 36 to clear" — and it has to
   * stand at the place that number names, or the label and the mark disagree by
   * the exact amount `points()` corrects. z* is 36.05 raw and 36 printed; the
   * shipped floor is 26.05 raw and 26 printed.
   */
  private markPct(z: number): number {
    return points(z, this.cfg);
  }

  /**
   * The window: one mark for the floor and the shaded band it opens up to the
   * notch. Built here rather than checked into index.html for the reason the 21
   * ticks are — the instrument draws its own face — and drawn AT ALL only on
   * the levels that run the check, so a level with no floor has no element to
   * mistake for one.
   *
   * Inserted BEFORE the notch, which puts the band over the moving wash and
   * under every mark that moves. Over the wash because a window that vanished
   * where the fill covers it would appear to start at the needle; under the
   * needle because the needle is what the player is asked to move and nothing
   * here may compete with it.
   */
  private buildWindow(): void {
    const floor = this.floorZ;
    if (floor === null) return;
    // `usableFloor` has already refused a floor that opens no window, so these
    // two are in order and the band has width.
    const from = this.markPct(floor);
    const to = this.markPct(this.cfg.zStar);
    const doc = this.els.meter.ownerDocument;
    const band = doc.createElement("div");
    band.className = "band";
    band.id = "band";
    band.style.left = `${from}%`;
    band.style.width = `${to - from}%`;
    const mark = doc.createElement("div");
    mark.className = "floormark";
    mark.id = "floormark";
    mark.style.left = `${from}%`;
    this.els.face.insertBefore(band, this.els.notch);
    this.els.face.insertBefore(mark, this.els.notch);
  }

  private measure(): void {
    this.faceWidth = this.els.face.clientWidth || 360;
    this.dpr = this.els.meter.ownerDocument.defaultView?.devicePixelRatio ?? 1;
  }

  /** Detent: quantise, then snap to whole device pixels so a 1px needle stays
   *  exactly 1px instead of antialiasing into two half columns whose apparent
   *  weight changes as it moves. */
  private toPx(z: number): number {
    const q = Math.round(this.pct(z) * 2) / 2;
    return Math.round((q / 100) * this.faceWidth * this.dpr) / this.dpr;
  }

  paint(z: number, instant: boolean): void {
    const { needle, fill } = this.els;
    if (instant) {
      needle.style.transitionDuration = "0ms";
      fill.style.transitionDuration = "0ms";
    }
    needle.style.transform = `translateX(${this.toPx(z)}px)`;
    fill.style.transform = `scaleX(${(Math.round(this.pct(z) * 2) / 2 / 100).toFixed(4)})`;
    if (instant) {
      void needle.offsetWidth; // flush before restoring durations
      needle.style.transitionDuration = "";
      fill.style.transitionDuration = "";
    }
  }

  /** Dim the number while a keystroke is outstanding. The needle does NOT move
   *  on a guess: no reading, no motion (§10.4 — never invent a value). */
  setPending(pending: boolean): void {
    this.els.num.setAttribute("data-pending", pending ? "1" : "0");
  }

  /**
   * Apply a real reading. Coalesced into the next animation frame: two readings
   * landing in the same frame paint once, at the newer value, so a superseded
   * reading never gets a frame of its own. That is the whole of "optimistic
   * within one frame, reconciled on the real score" — the tween retargets from
   * wherever the needle currently is rather than restarting.
   */
  setTarget(z: number): void {
    this.queued = z;
    const view = this.els.meter.ownerDocument.defaultView;
    if (this.frame !== null || typeof view?.requestAnimationFrame !== "function") {
      if (typeof view?.requestAnimationFrame !== "function") this.flush();
      return;
    }
    this.frame = view.requestAnimationFrame(() => {
      this.frame = null;
      this.flush();
    });
  }

  private flush(): void {
    if (this.queued === null) return;
    const z = this.queued;
    this.queued = null;
    this.apply(z);
  }

  private apply(z: number): void {
    const { needle, fill } = this.els;
    const span = this.cfg.scale.max - this.cfg.scale.min || 1;
    const delta = Math.abs(this.pct(z) - this.pct(this.last));
    const big = delta > 8;
    const reduced = this.reduceMotion?.matches === true;
    const dur = reduced ? "1ms" : big ? "380ms" : "160ms";
    const ease = big ? "cubic-bezier(.34,1.36,.64,1)" : "cubic-bezier(.4,0,.2,1)";
    needle.style.transitionDuration = dur;
    needle.style.transitionTimingFunction = ease;
    fill.style.transitionDuration = dur;
    fill.style.transitionTimingFunction = ease;
    this.paint(z, false);
    this.setPending(false);
    this.writeReadout(z);
    this.holdPeak(z);

    const below = z <= this.cfg.zStar;
    if (below && this.armed) {
      this.armed = false;
      this.cleared = true;
      this.els.live.textContent = this.cfg.copy.t("readout.valuetext_below", {
        z_display: formatPoints(z, this.cfg),
      });
      this.sweep();
      this.chime();
      this.onCross?.(true);
    } else if (!below && z > this.cfg.zStar + 0.02 * span) {
      // 2%-of-scale hysteresis, or a value bouncing across the line re-fires
      // the crossing moment on every keystroke.
      this.armed = true;
      if (this.cleared) {
        this.cleared = false;
        this.onCross?.(false);
      }
    }
    this.last = z;
    this.value = z;
  }

  private writeReadout(z: number): void {
    const below = z <= this.cfg.zStar;
    const flag = below ? "1" : "0";
    const { copy } = this.cfg;
    const display = formatPoints(z, this.cfg);
    this.els.num.textContent = display;
    // "Not detected", never "human": absence of a watermark does not prove a
    // person wrote it, and the readout must not claim more than the detector
    // can support (§10.7 rule 2).
    this.els.stateWord.textContent = copy.t(below ? "readout.below" : "readout.above");
    this.els.num.setAttribute("data-below", flag);
    this.els.stateWord.setAttribute("data-below", flag);
    this.els.fill.setAttribute("data-below", flag);
    this.els.notch.setAttribute("data-below", flag);
    this.els.tri.setAttribute("data-below", flag);
    this.els.floorLabel.setAttribute("data-below", flag);
    this.els.meter.setAttribute("aria-valuenow", display);
    // The floor is a mark on the face, and a mark on the face is no use to
    // anyone reading this instrument through a screen reader. The spoken value
    // carries it instead — the ONE place it can go without inventing a visual
    // cue that only some players get.
    //
    // The floor's own sentence REPLACES the line's rather than joining it. Both
    // are true down here — a floor that did not sit under the notch was refused
    // as a floor — but only one of them is the thing standing between the
    // player and the clear, and "26, below the line. Too clean. The needle
    // reads 26..." says the number twice to say less.
    const floor = this.floorZ;
    // `z < floor`, matching the gate's "at or above": the floor is IN the
    // window. The bound the player is told and the bound they are judged
    // against have to be the same bound.
    this.els.meter.setAttribute(
      "aria-valuetext",
      floor !== null && z < floor
        ? copy.t(`check.${FLOOR_CHECK}.reject`, {
            z_display: display,
            z_floor_display: formatPoints(floor, this.cfg),
          })
        : copy.t(below ? "readout.valuetext_below" : "readout.valuetext_above", {
            z_display: display,
          }),
    );
  }

  /** Peak-hold: rises instantly, decays linearly over 1200ms after 2s of
   *  stillness (§10.4). */
  private holdPeak(z: number): void {
    const { peak } = this.els;
    if (z > this.peak) {
      this.peak = z;
      peak.setAttribute("data-snap", "1");
      peak.style.transform = `translateX(${this.toPx(z)}px)`;
    }
    if (this.peakTimer !== null) clearTimeout(this.peakTimer);
    this.peakTimer = setTimeout(() => {
      this.peak = z;
      if (this.reduceMotion?.matches === true) peak.setAttribute("data-snap", "1");
      else peak.removeAttribute("data-snap");
      peak.style.transform = `translateX(${this.toPx(z)}px)`;
    }, 2000);
  }

  private sweep(): void {
    const el = this.els.sweep;
    if (this.reduceMotion?.matches === true) return;
    const animate = (el as { animate?: Element["animate"] }).animate;
    if (typeof animate !== "function") return;
    try {
      animate.call(
        el,
        [
          { transform: "translateX(0px)", opacity: 0.9 },
          { transform: `translateX(${this.faceWidth}px)`, opacity: 0 },
        ],
        { duration: 220, easing: "linear" },
      );
    } catch {
      /* decoration only */
    }
  }

  /** Audio must be armed by a user gesture; main.ts calls armAudio(). */
  armAudio(): void {
    if (this.audio !== null) return;
    const view = this.els.meter.ownerDocument.defaultView as
      | (Window & { AudioContext?: typeof AudioContext })
      | null;
    const Ctor = view?.AudioContext;
    if (typeof Ctor !== "function") return;
    try {
      this.audio = new Ctor();
    } catch {
      this.audio = null;
    }
  }

  setSound(on: boolean): void {
    this.soundOn = on;
  }

  private chime(): void {
    const ctx = this.audio;
    if (!this.soundOn || ctx === null) return;
    try {
      const t = ctx.currentTime;
      const notes: [number, number][] = [
        [528, 0.16],
        [792, 0.09],
      ];
      notes.forEach(([freq, gain], i) => {
        const o = ctx.createOscillator();
        const g = ctx.createGain();
        o.type = "sine";
        o.frequency.value = freq;
        g.gain.setValueAtTime(0, t);
        g.gain.linearRampToValueAtTime(gain, t + 0.012);
        g.gain.exponentialRampToValueAtTime(0.0001, t + 0.42);
        o.connect(g);
        g.connect(ctx.destination);
        o.start(t + i * 0.045);
        o.stop(t + 0.5);
      });
    } catch {
      /* a chime is never load-bearing */
    }
  }

  get current(): number {
    return this.value;
  }

  isBelow(): boolean {
    return this.value <= this.cfg.zStar;
  }
}
