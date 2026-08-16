"""launder-serve — the FastAPI app: static hosting, /api/detect, /api/submit, the gate.

Depends on `launder_core` only (TECH_PLAN.md §2.1). Nothing here may import
torch, transformers or anything else from `launder_forge`: this is the only
package installed into the Railway image, and the 250 MB budget is the reason
the three-package split exists at all.

Import order matters in exactly one place and it is stated twice on purpose:
**`main.py` mounts the API router BEFORE the static catch-all** (§11.4). A
`StaticFiles` mount at "/" swallows every path below it, so a router registered
after it is dead code that returns 404s in production and passes every test that
calls the router directly.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
