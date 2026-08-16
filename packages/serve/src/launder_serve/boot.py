"""`GET /` — the level page, rendered (TECH_PLAN.md §3, §5.4 step 1, §9.1).

**THIS IS THE FILE THAT WAS MISSING.** `web/index.html` has carried a "SERVER
TEMPLATE CONTRACT — launder_serve" comment block since M0, naming four regions
this module was supposed to rewrite, and nothing rewrote them: `_mount_static`
mounted `web/dist` verbatim, so every request got the checked-in DEV FIXTURE —
`"dev": true`, `"passage_id": "p_dev"`, `"assets": null`. Because `main.ts`
bails out of `upgrade()` when `assets` is falsy, the entire TypeScript detector,
the worker, the IndexedDB cache and the parity gate that guards them were dead
code at runtime, and the level's passage was never delivered to the client at all.

The four regions, and why each one is here rather than in a fetch:

1. `LAUNDER:BOOT` — the level, the detector expectation, the pristine reading,
   the generated primer demo, the intro's prefix_z, the asset URLs and
   `data/config/copy.toml` rendered to JSON. §9.1: "a first-time player makes
   ZERO API calls before playing."
2. `LAUNDER:TEXT` — the passage, inside the textarea. **The first character
   must follow the start tag immediately**: an HTML parser eats one newline
   directly after `<textarea>`, and the player would silently begin from a
   different string than the one the server scored — which then disagrees with
   `expected_z`, with `g_digest`, and with the scoreboard.
3. `#face` `--init-x` / `--init-n` — the needle and the fill, so the instrument
   reads `expected_z` before any network call and before any JS.
4. `#num` / `#stateword` / `data-below` — the same number in text, the right
   word next to it, and the below-the-line state on every element that is
   styled by it. The number is POINTS (§11), not z: `format_points` here and
   `formatPoints` in `web/src/game/needle.ts` must agree exactly, or the
   server-rendered first paint disagrees with the first client repaint — and
   `data-below` must reach the SAME SIX elements `needle.ts` marks, or the two
   paints disagree about which side of the line the reading is on.

Every substitution asserts it matched exactly once. A build whose markup drifted
must fail loudly here, not ship a page that silently renders the fixture again.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from launder_core.gates.feedback import CopyBook
from launder_core.schemas import LevelConfig, PassagePublic
from launder_serve.content import Content, PassageBundle
from launder_serve.engine import ServerDetector, text_hash

__all__ = [
    "BOOT_SCHEMA",
    "POINTS_MAX",
    "POINTS_MIN",
    "BootRenderer",
    "MissingRegion",
    "build_boot_payload",
    "format_points",
    "points",
    "render_index",
]

_log = logging.getLogger("launder.boot")

BOOT_SCHEMA = "launder.boot/2"

BOOT_OPEN = "<!-- LAUNDER:BOOT -->"
BOOT_CLOSE = "<!-- /LAUNDER:BOOT -->"
TEXT_OPEN = "<!-- LAUNDER:TEXT -->"
TEXT_CLOSE = "<!-- /LAUNDER:TEXT -->"

#: The INTERNAL scale of the instrument, in z (§10.5). Geometry, thresholds, the
#: wire and the DB are all still z; only the printed number is points.
SCALE_MIN = -2.0
SCALE_MAX = 10.0

#: The DISPLAYED scale (§11). z is a statistic nobody can place and -2..10 reads
#: as broken, so the readout is a monotone relabelling of z onto 0..100.
#: Mirrored in `index.html`'s `aria-valuemin`/`aria-valuemax`, which this module
#: rewrites from these two, and in `web/src/game/needle.ts`.
#:
#: It is NEVER rendered with a `%` sign and never called a percentage: the
#: standing rule for `[readout]` is that the product never shows a "% AI"
#: figure, and a number that looks like one would break it. This is the
#: "Watermark evidence" meter, not a probability.
POINTS_MIN = 0
POINTS_MAX = 100

#: `data/assets` filenames the browser needs for the LOCAL detector. Absent
#: files mean `assets: null`, which is M2 — the server-detect game, complete and
#: shippable, with the local path simply not offered (§5.4 step 5).
TOKENIZER_ASSET = "gemma3-tok.v1.bin.br"
TABLE_ASSET = "sampling_table.v1.bin"
THRESHOLDS_ASSET = "thresholds.v1.json"


class MissingRegion(RuntimeError):
    """`web/dist/index.html` does not carry a region this module must rewrite.

    A hard failure on purpose. The alternative is serving the checked-in dev
    fixture to real players, which is precisely the bug this module exists to
    fix and which produced a page whose every `/api/detect` 404'd.
    """


# ---------------------------------------------------------------------------
# escaping
# ---------------------------------------------------------------------------


def json_for_script(payload: Any) -> str:
    """JSON safe to embed in `<script type="application/json">`.

    `<` becomes `\\u003c`, NOT `&lt;`. A `script` element is a *raw text*
    element: character references inside it are not decoded, so `&lt;` would
    arrive at `JSON.parse` as the five literal characters and the copy would
    render with visible entities. `\\u003c` is valid JSON, decodes to `<`, and
    makes `</script>` unrepresentable in the payload.
    """
    out = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    # U+2028 LINE SEPARATOR and U+2029 PARAGRAPH SEPARATOR are valid inside a
    # JSON string and are line terminators to older JS parsers. Spelled with
    # `chr()` rather than as literals so the source carries no invisible
    # characters a reviewer cannot see.
    return out.replace(chr(0x2028), "\\u2028").replace(chr(0x2029), "\\u2029")


def escape_for_textarea(text: str) -> str:
    """`&` and `<` only. A textarea IS an escapable-raw-text element, so
    character references are decoded — which is why this escaping is both
    necessary and sufficient. Quotes must NOT be escaped: they are ordinary
    passage characters here and `&quot;` would change the scored text."""
    return text.replace("&", "&amp;").replace("<", "&lt;")


# ---------------------------------------------------------------------------
# the payload
# ---------------------------------------------------------------------------


def _level_wire(level: LevelConfig) -> dict[str, Any]:
    return {
        "id": level.id,
        "name": level.name,
        "teaches": level.teaches,
        "checks": [{"check": s.check, "params": dict(s.params)} for s in level.checks],
    }


def _reading_wire(detector: ServerDetector, text: str, passage: PassagePublic) -> dict[str, Any]:
    """The pristine reading, in the exact `/api/detect` response shape.

    `seq = 0` so the client's monotonic guard (§10.2) accepts anything the
    network or the worker produces afterwards.
    """
    readout = detector.read_tokens(text, passage)
    return {
        "seq": 0,
        "text_hash": text_hash(text),
        "score": readout.reading.score,
        "z": readout.reading.z,
        "z_star": readout.reading.z_star,
        "n_scored": readout.reading.n_scored,
        "n_tokens": readout.n_tokens,
        "tokens": [
            {"s": t.s, "e": t.e, "heat": t.heat, "masked": t.masked} for t in readout.tokens
        ],
        "preview_distance": 0,
    }


def _primer_demo(
    detector: ServerDetector, copy: CopyBook, passage: PassagePublic
) -> dict[str, Any] | None:
    """The primer's stained sentence, GENERATED (§10.7).

    "A hand-authored stain would be the one surface in the product free to teach
    the false heuristic, so it is forbidden." The heat here is the production
    detector's own contribution per token over the real key — which is why the
    cold slots are cold because the POSITION had no options, not because the
    words look plain. Returns `None` when the sentence is absent, and the primer
    then renders it unstained rather than inventing a stain.
    """
    raw: Any = copy.raw.get("primer", {})
    sentence = str(raw.get("demo", "")) if isinstance(raw, dict) else ""
    if not sentence:
        return None
    readout = detector.read_tokens(sentence, passage)
    return {
        "text": sentence,
        "tokens": [
            {"s": t.s, "e": t.e, "heat": t.heat, "masked": t.masked} for t in readout.tokens
        ],
    }


def _assets_wire(data_root: Path) -> dict[str, str] | None:
    assets = data_root / "assets"
    blob = assets / TOKENIZER_ASSET
    table = assets / TABLE_ASSET
    if not (blob.is_file() and table.is_file()):
        return None
    out = {
        "tokenizer_url": f"/data/assets/{TOKENIZER_ASSET}",
        "sampling_table_url": f"/data/assets/{TABLE_ASSET}",
    }
    if (assets / THRESHOLDS_ASSET).is_file():
        out["calibration_url"] = f"/data/assets/{THRESHOLDS_ASSET}"
    return out


def build_boot_payload(
    *,
    content: Content,
    detector: ServerDetector,
    bundle: PassageBundle,
    ruleset: LevelConfig,
    level_n: int,
    dev: bool,
) -> dict[str, Any]:
    """The `launder.boot/2` object `web/src/state.ts` parses. One place."""
    public = bundle.public
    reading = _reading_wire(detector, public.text, public)
    intro = (
        {
            "prefix_z": list(public.intro.prefix_z),
            "word_index": list(public.intro.word_index),
        }
        if public.intro is not None
        else None
    )
    return {
        "schema": BOOT_SCHEMA,
        "dev": dev,
        # The player's position in the campaign, and the total it is shown
        # against ("Level 3 of 15"). Both are rendered server-side into the rail
        # head by the client from these two numbers — there is no date, no
        # puzzle number and nothing that rolls over.
        "level_n": level_n,
        "level_count": content.level_count,
        "passage_id": public.id,
        "level": _level_wire(ruleset),
        "par": bundle.par,
        "asset_bundle_id": content.asset_bundle_id,
        "wm_config_id": content.wm_config_id,
        "detector": {
            # z* comes from the READING, not from a constant: `thresholds.v1.json`
            # is the authority and the level may select its own bucket set.
            "z_star": reading["z_star"],
            # The pristine passage's z, RECOMPUTED here rather than copied from
            # the packed `detector.expected_z`. If the two disagree the passage
            # was packed against different assets, and §4.5's tripwire is the
            # client comparing them — which only means something if this number
            # is the one this process actually computes.
            "expected_z": reading["z"],
            "scale": {"min": SCALE_MIN, "max": SCALE_MAX},
        },
        "reading": reading,
        "primer_demo": _primer_demo(detector, content.copy, public),
        "intro": intro,
        "assets": _assets_wire(content.data_root),
        "copy": content.copy.raw,
    }


# ---------------------------------------------------------------------------
# the injection
# ---------------------------------------------------------------------------


def _replace_region(html: str, open_tag: str, close_tag: str, inner: str, what: str) -> str:
    start = html.find(open_tag)
    end = html.find(close_tag)
    if start < 0 or end < 0 or end < start:
        raise MissingRegion(
            f"web/dist/index.html has no {what} region ({open_tag} ... {close_tag}). "
            "launder-serve rewrites it on every request; without the markers the page "
            "would ship the checked-in dev fixture, whose passage_id is `p_dev` and whose "
            "every /api/detect 404s. Restore the markers in web/index.html."
        )
    if html.count(open_tag) != 1 or html.count(close_tag) != 1:
        raise MissingRegion(
            f"web/dist/index.html carries {html.count(open_tag)} {open_tag} markers and "
            f"{html.count(close_tag)} {close_tag} markers; exactly one of each is required."
        )
    return html[: start + len(open_tag)] + inner + html[end:]


def _sub_once(
    html: str,
    pattern: str,
    replacement: str | Callable[[re.Match[str]], str],
    what: str,
) -> str:
    out, n = re.subn(pattern, replacement, html, count=2)
    if n != 1:
        raise MissingRegion(
            f"web/dist/index.html: the {what} substitution matched {n} times, expected "
            f"exactly 1 (pattern {pattern!r}). The markup drifted from the SERVER TEMPLATE "
            "CONTRACT block in web/index.html; fix one or the other rather than shipping a "
            "page whose instrument disagrees with its own reading."
        )
    return out


def _pct(z: float) -> float:
    """0..100 along the printed scale. The exact arithmetic of `Needle.pct`."""
    span = SCALE_MAX - SCALE_MIN or 1.0
    return max(0.0, min(100.0, ((z - SCALE_MIN) / span) * 100.0))


def _round_half_up(value: float) -> int:
    """`Math.round`, not Python's `round`.

    Python rounds halves to EVEN and JavaScript rounds them UP, so a z landing
    exactly on x.5 points would print 36 here and 37 in `needle.ts` — the
    server-rendered first paint disagreeing with the first client repaint by
    one, on one passage in a hundred, which is the hardest kind of disagreement
    to ever notice.
    """
    return math.floor(value + 0.5)


def points(z: float, z_star: float) -> int:
    """z relabelled onto 0..100 for display (§11). A monotone map, not a probability.

    The side of the line WINS OVER THE ROUNDING. Rounding can put a z that is
    above the notch onto the same integer as the notch itself, which is exactly
    the "2.3 on both sides" bug that the two-decimal z display was introduced to
    avoid — and a readout that disagrees with the verdict is the one
    disagreement this game cannot survive.
    """
    p_star = _round_half_up(_pct(z_star))
    p = max(POINTS_MIN, min(POINTS_MAX, _round_half_up(_pct(z))))
    if z > z_star and p <= p_star:
        p = min(p_star + 1, POINTS_MAX)
    if z <= z_star and p > p_star:
        p = p_star
    return p


def format_points(z: float, z_star: float) -> str:
    """The readout's text. A whole number, never a decimal and never a `%`."""
    return str(points(z, z_star))


def render_index(
    template: str,
    *,
    payload: dict[str, Any],
    passage_text: str,
) -> str:
    """Rewrite the four regions. Raises `MissingRegion` if any does not match."""
    expected_z = float(payload["detector"]["expected_z"])
    z_star = float(payload["detector"]["z_star"])
    below = expected_z <= z_star
    # The GEOMETRY is still z: `--init-x`/`--init-n` are the needle's position
    # along the -2..10 scale and are already percentages. Only the printed
    # number is points.
    pct = _pct(expected_z)
    readout = format_points(expected_z, z_star)

    html = _replace_region(
        template,
        BOOT_OPEN,
        BOOT_CLOSE,
        '\n<script type="application/json" id="launder-boot">\n'
        + json_for_script(payload)
        + "\n</script>\n",
        "boot payload",
    )

    # NOTE THE ABSENCE OF WHITESPACE. The first passage character follows the
    # start tag immediately; a newline here is eaten by the HTML parser and the
    # player would begin from a different string than the one that was scored.
    html = _replace_region(
        html,
        TEXT_OPEN,
        TEXT_CLOSE,
        _textarea_with(html, passage_text),
        "passage text",
    )

    html = _sub_once(
        html,
        r'(<div class="face" id="face" style=")[^"]*(")',
        lambda m: f"{m.group(1)}--init-x: {pct:.4f}%; --init-n: {pct / 100.0:.4f}{m.group(2)}",
        "#face --init-x/--init-n",
    )
    below_attr = ' data-below="1"' if below else ""
    html = _sub_once(
        html,
        r'(<span class="num" id="num" data-pending=")1(")(>)[^<]*(</span>)',
        lambda m: f"{m.group(1)}0{m.group(2)}{below_attr}{m.group(3)}{readout}{m.group(4)}",
        "#num readout",
    )
    html = _sub_once(
        html,
        r'(<span class="stateword" id="stateword" data-copy=")readout\.above(")',
        lambda m: f"{m.group(1)}readout.{'below' if below else 'above'}{m.group(2)}{below_attr}",
        "#stateword",
    )
    # ALL SIX ELEMENTS `needle.ts` marks, or none of them. `writeReadout()` sets
    # `data-below` on #num, #stateword, #fill, #notch, #tri and #floorlbl, and
    # rail.css styles `.num`, `.tri` and `.floorlbl` on that attribute — so a
    # server paint that marked only three of the six rendered the number in the
    # DETECTED colour beside the word "Not detected" until the first client
    # repaint. That one-frame contradiction is exactly what painting the state
    # server-side exists to prevent, so the two lists must not drift apart.
    if below:
        for element, pattern in (
            ("#fill", r'(<div class="fill" id="fill")(></div>)'),
            ("#notch", r'(<div class="notch" id="notch")(></div>)'),
            ("#tri", r'(<div class="tri" id="tri")(></div>)'),
            ("#floorlbl", r'(<div class="floorlbl" id="floorlbl")(></div>)'),
        ):
            html = _sub_once(html, pattern, r'\1 data-below="1"\2', element)
    # A screen reader is told the same number the sighted player is shown, on
    # the same scale: announcing z against a 0..100 meter would describe an
    # instrument nobody else can see.
    html = _sub_once(
        html,
        r'(aria-valuemin=")[^"]*(")',
        lambda m: f"{m.group(1)}{POINTS_MIN}{m.group(2)}",
        "meter aria-valuemin",
    )
    html = _sub_once(
        html,
        r'(aria-valuemax=")[^"]*(")',
        lambda m: f"{m.group(1)}{POINTS_MAX}{m.group(2)}",
        "meter aria-valuemax",
    )
    html = _sub_once(
        html,
        r'(aria-valuenow=")[^"]*(")',
        lambda m: f"{m.group(1)}{readout}{m.group(2)}",
        "meter aria-valuenow",
    )
    return html


def _textarea_with(html: str, passage_text: str) -> str:
    """Rebuild the textarea region, keeping the element's attributes verbatim."""
    start = html.find(TEXT_OPEN) + len(TEXT_OPEN)
    end = html.find(TEXT_CLOSE)
    region = html[start:end]
    open_end = region.find(">")
    close_start = region.rfind("</textarea")
    if open_end < 0 or close_start < 0:
        raise MissingRegion(
            "web/dist/index.html: the LAUNDER:TEXT region does not contain a complete "
            "<textarea ...>...</textarea>. The passage has nowhere to go."
        )
    return region[: open_end + 1] + escape_for_textarea(passage_text) + region[close_start:]


# ---------------------------------------------------------------------------
# the renderer the app holds
# ---------------------------------------------------------------------------


@dataclass
class _Cached:
    key: tuple[int, str, str, float]
    html: str


class BootRenderer:
    """Holds the template and caches the rendered page for each level.

    The template is re-read whenever `index.html`'s mtime changes, so `npm run
    dev`-style rebuilds are picked up without a restart, and the renders are
    cached because the pristine reading costs a tokenize plus a score and a
    level's passage never changes at all.

    The cache is a dict keyed by level rather than one slot: the campaign is
    linear and players are spread across all of it, so a single-entry cache
    would thrash on every other request and re-tokenize a passage per hit.
    """

    def __init__(self, dist: Path, content: Content, detector: ServerDetector) -> None:
        self.index = dist / "index.html"
        self.content = content
        self.detector = detector
        self._template: str | None = None
        self._template_mtime = -1.0
        self._cached: dict[int, _Cached] = {}

    def available(self) -> bool:
        return self.index.is_file()

    def missing_regions(self) -> list[str]:
        """Which of the four markers this template does not carry. Empty is good."""
        if not self.available():
            return [BOOT_OPEN, BOOT_CLOSE, TEXT_OPEN, TEXT_CLOSE]
        html = self.template()
        return [m for m in (BOOT_OPEN, BOOT_CLOSE, TEXT_OPEN, TEXT_CLOSE) if m not in html]

    def template(self) -> str:
        mtime = self.index.stat().st_mtime
        if self._template is None or mtime != self._template_mtime:
            self._template = self.index.read_text(encoding="utf-8")
            self._template_mtime = mtime
            self._cached.clear()
        return self._template

    def render(self, level_n: int) -> str | None:
        """The page for one level, or `None` when there is nothing to render.

        `None` means the campaign is empty — no packed passages and no dev
        fixture. An out-of-range `level_n` never reaches here: `main` resolves
        the request to a level that exists before asking for a render.
        """
        if not self.available():
            return None
        resolved = self.content.resolve(level_n)
        if resolved is None:
            return None
        bundle, ruleset = resolved
        template = self.template()
        key = (level_n, bundle.id, ruleset.id, self._template_mtime)
        cached = self._cached.get(level_n)
        if cached is not None and cached.key == key:
            return cached.html

        payload = build_boot_payload(
            content=self.content,
            detector=self.detector,
            bundle=bundle,
            ruleset=ruleset,
            level_n=level_n,
            dev=self.content.dev,
        )
        html = render_index(template, payload=payload, passage_text=bundle.public.text)
        self._cached[level_n] = _Cached(key=key, html=html)
        return html
