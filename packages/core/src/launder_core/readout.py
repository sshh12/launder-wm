"""The player-facing readout: z relabelled onto 0-100.

THE ONE DEFINITION. It lives in core, not in serve, because three things have to
agree on it or the screen contradicts itself: the server's first paint, the
browser's live needle, and the REJECTION MESSAGES the gate writes. It used to
live only in `launder_serve.boot`, which core cannot import — so the meter said
"50, and 36 to clear" while the rejection under it said "the needle reads 4.0
and has to reach 2.3". Same reading, two number systems, on one screen.

It is a monotone relabelling of z and nothing more. It is never rendered with a
`%`, never called a percentage, and never called a probability: the product does
not claim a likelihood it cannot support, and a 0-100 number that looks like one
would make that claim silently.
"""

from __future__ import annotations

import math

__all__ = [
    "POINTS_MAX",
    "POINTS_MIN",
    "SCALE_MAX",
    "SCALE_MIN",
    "format_points",
    "pct",
    "points",
]

#: The printed scale of the instrument, in z. Mirrored by `index.html`'s
#: aria-valuemin/max and by `web/src/game/needle.ts`.
SCALE_MIN = -2.0
SCALE_MAX = 10.0

POINTS_MIN = 0
POINTS_MAX = 100


def pct(z: float) -> float:
    """0..100 along the printed scale, UNROUNDED.

    The exact arithmetic of `Needle.pct` in the browser, and what the server
    writes into `--init-x`/`--init-n` so the instrument is already in the right
    place before any JavaScript runs.
    """
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
    p_star = _round_half_up(pct(z_star))
    p = max(POINTS_MIN, min(POINTS_MAX, _round_half_up(pct(z))))
    if z > z_star and p <= p_star:
        p = min(p_star + 1, POINTS_MAX)
    if z <= z_star and p > p_star:
        p = p_star
    return p


def format_points(z: float, z_star: float) -> str:
    """The readout's text. A whole number, never a decimal and never a `%`."""
    return str(points(z, z_star))
