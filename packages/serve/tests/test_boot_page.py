"""`GET /` renders index.html — TECH_PLAN.md §3, §5.4 step 1, §9.1.

THE REGRESSION THIS FILE EXISTS FOR: nothing rendered the page. `_mount_static`
mounted `web/dist` verbatim, so every request got the checked-in DEV FIXTURE
(`"dev": true`, `"passage_id": "p_dev"`, `"assets": null`). `main.ts` bails out
of `upgrade()` on a falsy `assets`, so the TypeScript detector, the worker, the
IndexedDB cache and the parity gate guarding them were all dead code at runtime,
and the daily passage was never delivered to the client at all. Every
`/api/detect` the page made 404'd on `p_dev`.

These tests use the REAL `web/index.html` as the template, because the thing
under test is the contract between that file's four marked regions and
`launder_serve.boot` — a hand-written fixture would test the fixture.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from launder_serve import boot as boot_mod
from launder_serve.boot import (
    BOOT_CLOSE,
    BOOT_OPEN,
    TEXT_CLOSE,
    TEXT_OPEN,
    BootRenderer,
    MissingRegion,
    escape_for_textarea,
    json_for_script,
)
from launder_serve.content import Content
from launder_serve.engine import CoreDetector
from launder_serve.main import create_app
from launder_serve.settings import Settings, repo_root


@pytest.fixture(scope="session")
def index_template() -> str:
    """The real `web/index.html`. `vite build` copies it through verbatim."""
    src = repo_root() / "web" / "index.html"
    if not src.is_file():
        pytest.skip("web/index.html is not present in this checkout")
    return src.read_text(encoding="utf-8")


@pytest.fixture
def dist(tmp_path: Path, index_template: str) -> Path:
    out = tmp_path / "dist"
    (out / "assets").mkdir(parents=True)
    (out / "index.html").write_text(index_template, encoding="utf-8")
    return out


@pytest.fixture
def renderer(dist: Path, content: Content) -> BootRenderer:
    return BootRenderer(dist, content, CoreDetector())


def _boot_of(html: str) -> dict[str, Any]:
    m = re.search(r'<script type="application/json" id="launder-boot">(.*?)</script>', html, re.S)
    assert m is not None, "no #launder-boot script in the rendered page"
    payload: dict[str, Any] = json.loads(m.group(1))
    return payload


def _textarea_of(html: str) -> str:
    region = html[html.index(TEXT_OPEN) : html.index(TEXT_CLOSE)]
    start = region.index(">", region.index("<textarea")) + 1
    end = region.rindex("</textarea")
    return region[start:end]


# ---------------------------------------------------------------------------
# the payload
# ---------------------------------------------------------------------------


def test_the_rendered_page_is_not_the_dev_fixture(
    renderer: BootRenderer, scheduled_day: Any
) -> None:
    html = renderer.render(scheduled_day.date)
    assert html is not None
    boot = _boot_of(html)
    assert boot["schema"] == "launder.boot/1"
    assert boot["dev"] is False
    assert boot["passage_id"] == scheduled_day.passage_id
    assert boot["level"]["id"] == scheduled_day.level_id
    assert boot["day"] == scheduled_day.date.isoformat()
    # The checked-in fixture's tells, all gone.
    assert boot["asset_bundle_id"] != ""
    assert boot["wm_config_id"].startswith("wm1:")


def test_the_boot_payload_names_the_local_detector_assets(
    renderer: BootRenderer, scheduled_day: Any
) -> None:
    """`assets: null` is what killed the LOCAL path.

    `main.ts`: `if (!assets?.tokenizer_url || !assets.sampling_table_url) return;`
    — with the fixture's null the state machine never left SERVER, so the whole
    TS detector was unreachable in production.
    """
    boot = _boot_of(renderer.render(scheduled_day.date) or "")
    assets = boot["assets"]
    assert assets is not None
    assert assets["tokenizer_url"] == "/data/assets/gemma3-tok.v1.bin.br"
    assert assets["sampling_table_url"] == "/data/assets/sampling_table.v1.bin"
    assert assets["calibration_url"] == "/data/assets/thresholds.v1.json"


def test_the_pristine_reading_is_inlined_so_the_first_paint_costs_no_api_call(
    renderer: BootRenderer, scheduled_day: Any, content: Content
) -> None:
    """§9.1: 'a first-time player makes ZERO API calls before playing'."""
    boot = _boot_of(renderer.render(scheduled_day.date) or "")
    bundle = content.passage(scheduled_day.passage_id)
    assert bundle is not None
    reading = boot["reading"]
    assert reading is not None
    assert reading["seq"] == 0
    assert len(reading["tokens"]) == reading["n_tokens"] > 0
    assert reading["n_scored"] <= reading["n_tokens"]
    # Recomputed here, not copied from the packed expectation: §4.5's tripwire
    # is the client comparing the two, which means nothing if the server just
    # echoes the file.
    direct = CoreDetector().read(bundle.public.text, bundle.public)
    assert reading["z"] == pytest.approx(direct.z, abs=0.0)
    assert boot["detector"]["expected_z"] == pytest.approx(direct.z, abs=0.0)
    assert boot["detector"]["z_star"] == pytest.approx(direct.z_star, abs=0.0)


def test_the_primer_demo_is_generated_not_hand_painted(
    renderer: BootRenderer, scheduled_day: Any, content: Content
) -> None:
    """§10.7: 'GENERATED, never hand-painted ... it is forbidden'."""
    boot = _boot_of(renderer.render(scheduled_day.date) or "")
    demo = boot["primer_demo"]
    assert demo is not None
    assert demo["text"] == content.copy.raw["primer"]["demo"]
    assert len(demo["tokens"]) > 5
    heat = [t["heat"] for t in demo["tokens"]]
    # A real detector reading varies; a hand-painted one is whatever somebody
    # typed. Constant heat would mean the sentence never reached the detector.
    assert len(set(heat)) > 3


def test_the_primer_demo_teaches_the_RIGHT_heuristic(
    renderer: BootRenderer, scheduled_day: Any, content: Content
) -> None:
    """§10.7's UNGUARDED INVARIANT, now guarded.

    > "Whatever passage ships here, assert that property in CI — pick a sentence
    >  where at least one function word is in the top heat quartile and at least
    >  one content word is below the [cutoff]."

    That assertion existed nowhere: a grep for "quartile" across the repo
    returned nothing. It is the §4.6 guardrail made checkable — the demo strip
    is the one surface in the product that could teach "spot the fancy AI word",
    and the defence is that on the real detector the most banal words are often
    the hottest and content words are often cold. A candidate sentence that
    fails this is the WRONG SENTENCE, and the failure has to be a red test
    rather than a thing somebody notices later.
    """
    from launder_core.gates.checks.close_paraphrase import stopwords

    boot = _boot_of(renderer.render(scheduled_day.date) or "")
    demo = boot["primer_demo"]
    assert demo is not None, "the primer would render its sentence unstained"
    text = demo["text"]
    tokens = [
        (text[t["s"] : t["e"]].strip().strip(".,;:!?").lower(), t["heat"]) for t in demo["tokens"]
    ]
    tokens = [(w, h) for w, h in tokens if w]
    heats = sorted(h for _, h in tokens)
    n = len(heats)
    q1 = heats[int(0.25 * (n - 1))]
    q3 = heats[int(0.75 * (n - 1))]
    stops = stopwords()

    function_words_hot = [w for w, h in tokens if h >= q3 and w in stops]
    content_words_cold = [w for w, h in tokens if h <= q1 and w not in stops]
    assert function_words_hot, (
        f"no function word reaches the top heat quartile in {text!r}. The demo would "
        "teach that heat tracks fancy vocabulary, which is the one heuristic "
        "CONCEPT.md forbids. Pick a different sentence for [primer].demo."
    )
    assert content_words_cold, (
        f"no content word falls in the bottom heat quartile in {text!r}. Same problem, "
        "other direction."
    )


def test_the_copy_tree_is_inlined(renderer: BootRenderer, scheduled_day: Any) -> None:
    """§10.7: no player-facing string lives in web/src/, so all of copy.toml ships."""
    boot = _boot_of(renderer.render(scheduled_day.date) or "")
    assert boot["copy"]["readout"]["above"]
    assert boot["copy"]["check"]["detector_threshold"]["label"]


# ---------------------------------------------------------------------------
# the passage text
# ---------------------------------------------------------------------------


def test_the_passage_is_in_the_textarea_byte_for_byte(
    renderer: BootRenderer, scheduled_day: Any, content: Content
) -> None:
    bundle = content.passage(scheduled_day.passage_id)
    assert bundle is not None
    html = renderer.render(scheduled_day.date) or ""
    assert _textarea_of(html) == escape_for_textarea(bundle.public.text)


def test_no_whitespace_follows_the_textarea_start_tag(
    renderer: BootRenderer, scheduled_day: Any, content: Content
) -> None:
    """An HTML parser eats ONE newline directly after `<textarea>`.

    If the server emits one, the player starts from a different string than the
    one that was scored — and then `expected_z`, `g_digest` and the scoreboard
    all describe a text nobody is editing.
    """
    bundle = content.passage(scheduled_day.passage_id)
    assert bundle is not None
    body = _textarea_of(renderer.render(scheduled_day.date) or "")
    assert not body[:1].isspace()
    assert body.startswith(bundle.public.text[:20])


def test_textarea_escaping_is_ampersand_and_lt_only() -> None:
    # A textarea is an escapable-raw-text element: `&amp;`/`&lt;` are decoded,
    # so they are necessary. Quotes are ordinary passage characters and escaping
    # them would change the SCORED text.
    assert escape_for_textarea('a & b < c "d"') == 'a &amp; b &lt; c "d"'


def test_the_boot_json_escapes_lt_as_a_unicode_escape_not_an_entity() -> None:
    """A `script` element is RAW TEXT: `&lt;` inside it is not decoded.

    Escaping as an entity would put five literal characters into the copy the
    player reads; `\\u003c` is valid JSON and makes `</script>` unrepresentable.
    """
    out = json_for_script({"t": "</script><b>&amp;"})
    assert "</script>" not in out
    assert "&lt;" not in out
    assert json.loads(out)["t"] == "</script><b>&amp;"


# ---------------------------------------------------------------------------
# the instrument, before any JS
# ---------------------------------------------------------------------------


def test_the_needle_and_the_readout_are_correct_before_any_js(
    renderer: BootRenderer, scheduled_day: Any
) -> None:
    """§5.4 step 1: correct before any network call AND before any JS."""
    html = renderer.render(scheduled_day.date) or ""
    boot = _boot_of(html)
    z = boot["detector"]["expected_z"]
    pct = max(
        0.0,
        min(100.0, ((z - boot_mod.SCALE_MIN) / (boot_mod.SCALE_MAX - boot_mod.SCALE_MIN)) * 100.0),
    )

    face = re.search(r"--init-x: ([0-9.]+)%; --init-n: ([0-9.]+)", html)
    assert face is not None
    assert float(face.group(1)) == pytest.approx(pct, abs=1e-3)
    assert float(face.group(2)) == pytest.approx(pct / 100.0, abs=1e-3)

    num = re.search(r'<span class="num" id="num" data-pending="(\d)">([^<]*)</span>', html)
    assert num is not None
    assert num.group(1) == "0", "the readout is no longer pending: the server computed it"
    assert num.group(2) == f"{z:.2f}"
    assert f'aria-valuenow="{z:.2f}"' in html


def test_the_stateword_matches_the_side_of_the_line_the_needle_is_on(
    renderer: BootRenderer, scheduled_day: Any
) -> None:
    html = renderer.render(scheduled_day.date) or ""
    boot = _boot_of(html)
    below = boot["detector"]["expected_z"] <= boot["detector"]["z_star"]
    want = "readout.below" if below else "readout.above"
    assert f'id="stateword" data-copy="{want}"' in html
    assert ('data-below="1"' in html) is below


# ---------------------------------------------------------------------------
# loud failure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("marker", [BOOT_OPEN, BOOT_CLOSE, TEXT_OPEN, TEXT_CLOSE])
def test_a_template_missing_a_region_raises_instead_of_shipping_the_fixture(
    dist: Path, content: Content, index_template: str, marker: str
) -> None:
    (dist / "index.html").write_text(index_template.replace(marker, ""), encoding="utf-8")
    renderer = BootRenderer(dist, content, CoreDetector())
    with pytest.raises(MissingRegion):
        renderer.render(date(2026, 9, 1))


def test_drifted_markup_raises_rather_than_rendering_a_wrong_instrument(
    dist: Path, content: Content, index_template: str
) -> None:
    broken = index_template.replace('<span class="num" id="num" data-pending="1">', "<span>")
    (dist / "index.html").write_text(broken, encoding="utf-8")
    renderer = BootRenderer(dist, content, CoreDetector())
    with pytest.raises(MissingRegion, match="#num readout"):
        renderer.render(date(2026, 9, 1))


# ---------------------------------------------------------------------------
# over HTTP, with the static catch-all mounted (the mount-order regression)
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_slash_renders_and_does_not_fall_through_to_static(
    dist: Path,
    settings: Settings,
    content: Content,
    repos: Any,
    scheduled_day: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("launder_serve.main.today_utc", lambda: scheduled_day.date)
    app = create_app(
        settings=settings.model_copy(update={"web_dist_dir": dist}),
        content=content,
        repos=repos,
        mount_static=True,
    )
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            for path in ("/", "/index.html"):
                r = await http.get(path)
                assert r.status_code == 200, path
                assert r.headers["content-type"].startswith("text/html")
                assert r.headers["cache-control"] == "no-cache"
                boot = _boot_of(r.text)
                assert boot["passage_id"] == scheduled_day.passage_id, (
                    f"{path} fell through to the static mount and served the dev fixture"
                )
                assert boot["dev"] is False
