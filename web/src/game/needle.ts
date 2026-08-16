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
 */

import type { Copy } from "../state";

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
}

/** Two decimals: z* is 2.3263, and one decimal would print "2.3" on both sides
 *  of the line. The readout must never disagree with the verdict. */
export function formatZ(z: number): string {
  return z.toFixed(2);
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
  onCross: ((below: boolean) => void) | null = null;

  constructor(els: NeedleElements, cfg: NeedleConfig, initialZ: number) {
    this.els = els;
    this.cfg = cfg;
    this.value = initialZ;
    this.last = initialZ;
    this.peak = initialZ;
    const view = els.meter.ownerDocument.defaultView;
    this.reduceMotion =
      typeof view?.matchMedia === "function"
        ? view.matchMedia("(prefers-reduced-motion: reduce)")
        : null;
    this.buildScale();
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
    const { min, max } = this.cfg.scale;
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
      const z = min + ((max - min) * p) / 100;
      s.textContent = Number.isInteger(z) ? String(z) : z.toFixed(1);
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
    const at = `${this.pct(this.cfg.zStar)}%`;
    this.els.notch.style.left = at;
    this.els.tri.style.left = at;
    this.els.floorLabel.style.left = at;
    this.els.floorLabel.style.transform = "translateX(-50%)";
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
        z_display: formatZ(z),
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
    this.els.num.textContent = formatZ(z);
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
    this.els.meter.setAttribute("aria-valuenow", formatZ(z));
    this.els.meter.setAttribute(
      "aria-valuetext",
      copy.t(below ? "readout.valuetext_below" : "readout.valuetext_above", {
        z_display: formatZ(z),
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
