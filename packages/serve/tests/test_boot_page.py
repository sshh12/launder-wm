"""`GET /` renders index.html — TECH_PLAN.md §3, §5.4 step 1, §9.1.

THE REGRESSION THIS FILE EXISTS FOR: nothing rendered the page. `_mount_static`
mounted `web/dist` verbatim, so every request got the checked-in DEV FIXTURE
(`"dev": true`, `"passage_id": "p_dev"`, `"assets": null`). `main.ts` bails out
of `upgrade()` on a falsy `assets`, so the TypeScript detector, the worker, the
IndexedDB cache and the parity gate guarding them were all dead code at runtime,
and the level's passage was never delivered to the client at all. Every
`/api/detect` the page made 404'd on `p_dev`.

These tests use the REAL `web/index.html` as the template, because the thing
under test is the contract between that file's four marked regions and
`launder_serve.boot` — a hand-written fixture would test the fixture.
"""

from __future__ import annotations

import json
import re
import shutil
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
from launder_serve.content import DEV_PASSAGE_ID, Content, load_content
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
    renderer: BootRenderer, level_n: int, passage_id: str, level_id: str
) -> None:
    html = renderer.render(level_n)
    assert html is not None
    boot = _boot_of(html)
    assert boot["schema"] == "launder.boot/2"
    assert boot["dev"] is False
    assert boot["passage_id"] == passage_id
    assert boot["level"]["id"] == level_id
    # The checked-in fixture's tells, all gone.
    assert boot["asset_bundle_id"] != ""
    assert boot["wm_config_id"].startswith("wm1:")


def test_the_boot_payload_carries_the_players_position_in_the_campaign(
    renderer: BootRenderer, content: Content, level_n: int
) -> None:
    """`level_n` and `level_count` are the whole of "Level 3 of 15".

    They replace `day` and `puzzle_number` outright: there is no date on the
    wire any more, and nothing rolls over. A payload still carrying either
    would mean the client renders a date it can no longer be given.
    """
    boot = _boot_of(renderer.render(level_n) or "")
    assert boot["level_n"] == level_n
    assert boot["level_count"] == content.level_count
    assert 1 <= boot["level_n"] <= boot["level_count"]
    assert "day" not in boot
    assert "puzzle_number" not in boot


def test_each_level_renders_its_own_passage(renderer: BootRenderer, content: Content) -> None:
    """The cache is per level, not one slot: rendering level 2 must not hand
    back the page level 1 was rendered into."""
    for entry in content.campaign[:3]:
        boot = _boot_of(renderer.render(entry.n) or "")
        assert boot["level_n"] == entry.n
        assert boot["passage_id"] == entry.passage_id
        assert boot["level"]["id"] == entry.level_id


def test_a_level_past_the_end_of_the_campaign_renders_nothing(
    renderer: BootRenderer, content: Content
) -> None:
    """`render` is not a clamp. `main` resolves the request to a level that
    exists first, and anything else is a bug that must not be papered over by
    quietly serving level 1."""
    assert renderer.render(content.level_count + 1) is None
    assert renderer.render(0) is None


def test_a_clone_with_no_packed_passages_still_renders_the_dev_fixture(
    tmp_path: Path, dist: Path, real_data_root: Path
) -> None:
    """A fresh clone has an EMPTY `data/passages/` and must still be playable.

    It cannot be filled without a GPU and the gated Gemma-3 weights, so the dev
    fixture stands in for the whole campaign — level 1 of 1, `dev: true`. This
    is the one state that is NOT a hole in the campaign, and it is why an empty
    passage set does not trip `strict`.
    """
    if not (real_data_root / "dev" / "passage.txt").is_file():
        pytest.skip("data/dev/passage.txt is not present in this checkout")
    root = tmp_path / "bare"
    (root / "passages").mkdir(parents=True)
    shutil.copytree(real_data_root / "config", root / "config")
    shutil.copytree(real_data_root / "assets", root / "assets")
    shutil.copytree(real_data_root / "dev", root / "dev")

    bare = load_content(root, is_production=False)
    assert bare.dev is True
    assert bare.level_count == 1
    assert bare.campaign[0].passage_id == DEV_PASSAGE_ID

    boot = _boot_of(BootRenderer(dist, bare, CoreDetector()).render(1) or "")
    assert boot["dev"] is True
    assert boot["passage_id"] == DEV_PASSAGE_ID
    assert boot["level_n"] == 1
    assert boot["level_count"] == 1


def test_the_boot_payload_names_the_local_detector_assets(
    renderer: BootRenderer, level_n: int
) -> None:
    """`assets: null` is what killed the LOCAL path.

    `main.ts`: `if (!assets?.tokenizer_url || !assets.sampling_table_url) return;`
    — with the fixture's null the state machine never left SERVER, so the whole
    TS detector was unreachable in production.
    """
    boot = _boot_of(renderer.render(level_n) or "")
    assets = boot["assets"]
    assert assets is not None
    assert assets["tokenizer_url"] == "/data/assets/gemma3-tok.v1.bin.br"
    assert assets["sampling_table_url"] == "/data/assets/sampling_table.v1.bin"
    assert assets["calibration_url"] == "/data/assets/thresholds.v1.json"


def test_the_pristine_reading_is_inlined_so_the_first_paint_costs_no_api_call(
    renderer: BootRenderer, level_n: int, passage_id: str, content: Content
) -> None:
    """§9.1: 'a first-time player makes ZERO API calls before playing'."""
    boot = _boot_of(renderer.render(level_n) or "")
    bundle = content.passage(passage_id)
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
    renderer: BootRenderer, level_n: int, passage_id: str, content: Content
) -> None:
    """§10.7: 'GENERATED, never hand-painted ... it is forbidden'."""
    boot = _boot_of(renderer.render(level_n) or "")
    demo = boot["primer_demo"]
    assert demo is not None
    assert demo["text"] == content.copy.raw["primer"]["demo"]
    assert len(demo["tokens"]) > 5
    heat = [t["heat"] for t in demo["tokens"]]
    # A real detector reading varies; a hand-painted one is whatever somebody
    # typed. Constant heat would mean the sentence never reached the detector.
    assert len(set(heat)) > 3


def test_the_primer_demo_teaches_the_RIGHT_heuristic(
    renderer: BootRenderer, level_n: int, passage_id: str, content: Content
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

    boot = _boot_of(renderer.render(level_n) or "")
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


def test_the_copy_tree_is_inlined(renderer: BootRenderer, level_n: int) -> None:
    """§10.7: no player-facing string lives in web/src/, so all of copy.toml ships."""
    boot = _boot_of(renderer.render(level_n) or "")
    assert boot["copy"]["readout"]["above"]
    assert boot["copy"]["check"]["detector_threshold"]["label"]


# ---------------------------------------------------------------------------
# the passage text
# ---------------------------------------------------------------------------


def test_the_passage_is_in_the_textarea_byte_for_byte(
    renderer: BootRenderer, level_n: int, passage_id: str, content: Content
) -> None:
    bundle = content.passage(passage_id)
    assert bundle is not None
    html = renderer.render(level_n) or ""
    assert _textarea_of(html) == escape_for_textarea(bundle.public.text)


def test_no_whitespace_follows_the_textarea_start_tag(
    renderer: BootRenderer, level_n: int, passage_id: str, content: Content
) -> None:
    """An HTML parser eats ONE newline directly after `<textarea>`.

    If the server emits one, the player starts from a different string than the
    one that was scored — and then `expected_z`, `g_digest` and the scoreboard
    all describe a text nobody is editing.
    """
    bundle = content.passage(passage_id)
    assert bundle is not None
    body = _textarea_of(renderer.render(level_n) or "")
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
    renderer: BootRenderer, level_n: int
) -> None:
    """§5.4 step 1: correct before any network call AND before any JS."""
    html = renderer.render(level_n) or ""
    boot = _boot_of(html)
    z = boot["detector"]["expected_z"]
    z_star = boot["detector"]["z_star"]
    pct = max(
        0.0,
        min(100.0, ((z - boot_mod.SCALE_MIN) / (boot_mod.SCALE_MAX - boot_mod.SCALE_MIN)) * 100.0),
    )

    # The GEOMETRY is still z: the needle sits at the same place on the -2..10
    # scale it always did. Only the printed number changed.
    face = re.search(r"--init-x: ([0-9.]+)%; --init-n: ([0-9.]+)", html)
    assert face is not None
    assert float(face.group(1)) == pytest.approx(pct, abs=1e-3)
    assert float(face.group(2)) == pytest.approx(pct / 100.0, abs=1e-3)

    # `[^>]*` because a below-the-line paint also carries `data-below="1"` here.
    num = re.search(r'<span class="num" id="num" data-pending="(\d)"[^>]*>([^<]*)</span>', html)
    assert num is not None
    assert num.group(1) == "0", "the readout is no longer pending: the server computed it"
    assert num.group(2) == str(boot_mod.points(z, z_star))
    assert f'aria-valuenow="{boot_mod.points(z, z_star)}"' in html


def test_the_readout_is_a_whole_number_of_points_on_a_0_100_meter(
    renderer: BootRenderer, level_n: int
) -> None:
    """§11: z is a statistic nobody can place and -2..10 reads as broken.

    The printed number is points, and the meter it is announced against says so
    — a screen reader told "6.44 out of -2 to 10" while the screen says "70" is
    describing an instrument nobody else can see.
    """
    html = renderer.render(level_n) or ""
    assert 'aria-valuemin="0"' in html
    assert 'aria-valuemax="100"' in html
    num = re.search(r'<span class="num" id="num" data-pending="0"[^>]*>([^<]*)</span>', html)
    assert num is not None
    printed = num.group(1)
    assert printed.isdigit()
    assert 0 <= int(printed) <= 100
    # Never a percentage: the product does not show a "% AI" figure, and a
    # number carrying a % sign would read as exactly that.
    assert "%" not in printed


@pytest.mark.parametrize(
    "z,expected",
    [
        (-2.0, 0),  # the bottom of the scale
        (10.0, 100),  # the top
        (-99.0, 0),  # clamped, not negative
        (99.0, 100),  # clamped, not 842
        (2.3263, 36),  # the notch itself
        (6.44, 70),
    ],
)
def test_points_is_the_documented_relabelling_of_z(z: float, expected: int) -> None:
    assert boot_mod.points(z, 2.3263) == expected


def test_the_readout_never_disagrees_with_the_verdict() -> None:
    """The side of the line WINS OVER THE ROUNDING.

    z* = 2.3263 rounds to 36 points, and so does every z from about 2.28 to
    2.40 — including values ABOVE the notch. Printing 36 for one of those is
    the "2.3 on both sides" bug that the two-decimal z display was introduced
    to avoid, one relabelling later.
    """
    z_star = 2.3263
    at_the_line = boot_mod.points(z_star, z_star)
    just_above = boot_mod.points(z_star + 0.0001, z_star)
    assert just_above > at_the_line
    # ...and the whole neighbourhood above the notch stays above it.
    for i in range(1, 60):
        z = z_star + i * 0.001
        assert boot_mod.points(z, z_star) > at_the_line, z
    for i in range(1, 60):
        z = z_star - i * 0.001
        assert boot_mod.points(z, z_star) <= at_the_line, z


def test_the_stateword_matches_the_side_of_the_line_the_needle_is_on(
    renderer: BootRenderer, level_n: int
) -> None:
    html = renderer.render(level_n) or ""
    boot = _boot_of(html)
    below = boot["detector"]["expected_z"] <= boot["detector"]["z_star"]
    want = "readout.below" if below else "readout.above"
    assert f'id="stateword" data-copy="{want}"' in html
    assert ('data-below="1"' in html) is below


#: Every element `needle.ts`'s `writeReadout()` marks with `data-below`. The
#: server's first paint has to mark the same set: `rail.css` styles `.num`,
#: `.tri` and `.floorlbl` on the attribute too, so marking a subset paints a
#: contradiction until the first client repaint.
BELOW_MARKED_IDS = ("num", "stateword", "fill", "notch", "tri", "floorlbl")


@pytest.mark.parametrize("below", [True, False])
def test_below_the_line_paints_every_element_the_client_paints(
    index_template: str, below: bool
) -> None:
    """THE ONE-FRAME LIE. `boot.py` marked #stateword, #fill and #notch only.

    `needle.ts` marks six, and rail.css restyles `.num`, `.tri` and `.floorlbl`
    on `data-below` — so a passage that starts BELOW the line was served with the
    number still in the detected colour, sitting next to the words "Not
    detected", until JS ran. Painting the state server-side exists precisely to
    stop that frame, so a partial paint is worse than none: it is the same lie,
    harder to spot.
    """
    z_star = 2.3263
    z = z_star - 1.0 if below else z_star + 1.0
    html = boot_mod.render_index(
        index_template,
        payload={"detector": {"expected_z": z, "z_star": z_star}},
        passage_text="Some passage text.",
    )
    for element_id in BELOW_MARKED_IDS:
        marked = re.search(rf'id="{element_id}"[^>]*data-below="1"', html) is not None
        assert marked is below, f"#{element_id} data-below={marked}, expected {below}"


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
        renderer.render(1)


def test_drifted_markup_raises_rather_than_rendering_a_wrong_instrument(
    dist: Path, content: Content, index_template: str
) -> None:
    broken = index_template.replace('<span class="num" id="num" data-pending="1">', "<span>")
    (dist / "index.html").write_text(broken, encoding="utf-8")
    renderer = BootRenderer(dist, content, CoreDetector())
    with pytest.raises(MissingRegion, match="#num readout"):
        renderer.render(1)


# ---------------------------------------------------------------------------
# over HTTP, with the static catch-all mounted (the mount-order regression)
# ---------------------------------------------------------------------------


@pytest.fixture
def page_app(dist: Path, settings: Settings, content: Content, repos: Any) -> Any:
    return create_app(
        settings=settings.model_copy(update={"web_dist_dir": dist}),
        content=content,
        repos=repos,
        mount_static=True,
    )


@pytest.fixture
async def page_client(page_app: Any) -> Any:
    async with page_app.router.lifespan_context(page_app):
        transport = httpx.ASGITransport(app=page_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            yield http


@pytest.mark.anyio
async def test_get_slash_renders_and_does_not_fall_through_to_static(
    page_client: httpx.AsyncClient, content: Content
) -> None:
    for path in ("/", "/index.html"):
        r = await page_client.get(path)
        assert r.status_code == 200, path
        assert r.headers["content-type"].startswith("text/html")
        boot = _boot_of(r.text)
        assert boot["passage_id"] == content.campaign[0].passage_id, (
            f"{path} fell through to the static mount and served the dev fixture"
        )
        assert boot["dev"] is False


@pytest.mark.anyio
async def test_a_visitor_with_no_cookie_gets_level_one(
    page_client: httpx.AsyncClient, content: Content
) -> None:
    boot = _boot_of((await page_client.get("/")).text)
    assert boot["level_n"] == 1
    assert boot["level_count"] == content.level_count


@pytest.mark.anyio
async def test_the_cookie_selects_the_level_with_no_api_call(
    page_client: httpx.AsyncClient, content: Content
) -> None:
    """The whole reason `launder_level` exists: the returning player's level is
    in the FIRST paint, not one fetch and one reflow later."""
    r = await page_client.get("/", headers={"Cookie": "launder_level=4"})
    boot = _boot_of(r.text)
    assert boot["level_n"] == 4
    assert boot["passage_id"] == content.campaign[3].passage_id


@pytest.mark.anyio
async def test_the_query_parameter_beats_the_cookie(page_client: httpx.AsyncClient) -> None:
    """`?level=` is the test escape hatch: it renders that level regardless of
    progress, and it does not write the cookie."""
    r = await page_client.get("/", params={"level": "2"}, headers={"Cookie": "launder_level=5"})
    assert _boot_of(r.text)["level_n"] == 2
    assert "set-cookie" not in r.headers


@pytest.mark.anyio
@pytest.mark.parametrize("value", ["0", "-3", "999", "banana", "", "2.5"])
async def test_an_unusable_level_falls_back_to_one(
    page_client: httpx.AsyncClient, value: str
) -> None:
    """Not clamped into range: a typo'd `?level=150` rendering level 15 would
    make a test meaning "that level does not exist" pass against the wrong
    page."""
    assert _boot_of((await page_client.get("/", params={"level": value})).text)["level_n"] == 1
    cookied = await page_client.get("/", headers={"Cookie": f"launder_level={value}"})
    assert _boot_of(cookied.text)["level_n"] == 1


@pytest.mark.anyio
async def test_the_page_is_private_and_varies_on_the_cookie(
    page_client: httpx.AsyncClient,
) -> None:
    """A shared cache handing one player's level to the next visitor would drop
    them into somebody else's campaign, and a cache keyed on the URL alone would
    serve level 1 to everyone who arrived after the first visitor."""
    r = await page_client.get("/")
    assert r.headers["cache-control"] == "private, no-cache"
    assert r.headers["vary"] == "Cookie, Accept-Encoding"
