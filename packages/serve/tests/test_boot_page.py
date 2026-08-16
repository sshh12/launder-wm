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
    render_origin,
    safe_origin,
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


def test_the_boot_payload_carries_the_locked_phrases_so_the_rule_can_be_live(
    renderer: BootRenderer, content: Content
) -> None:
    """`locked_phrase` is answerable in the browser, but only with the phrases.

    Without them the level's one distinguishing rule is invisible until a
    submission fails: nothing on screen moves when the player breaks the phrase,
    and the checklist says "Verbatim phrase" without ever naming which. They are
    not a secret — each is a literal substring of the passage in the textarea,
    and the rejection message renders it back verbatim.
    """
    for entry in content.campaign:
        resolved = content.resolve(entry.n)
        assert resolved is not None
        bundle, ruleset = resolved
        boot = _boot_of(renderer.render(entry.n) or "")
        assert boot["locked_phrases"] == list(bundle.public.rules.locked_phrases)
        if any(spec.check == "locked_phrase" for spec in ruleset.checks):
            assert boot["locked_phrases"], (
                f"level {entry.n} runs locked_phrase against {bundle.id}, which declares no "
                "phrases; the gate would raise GateDataError and the browser would have "
                "nothing to check"
            )


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
# the fifth region: where the page says it lives
# ---------------------------------------------------------------------------


def _tag_value(html: str, pattern: str) -> str:
    m = re.search(pattern, html)
    assert m is not None, f"no match for {pattern}"
    return m.group(1)


CANONICAL = r'<link rel="canonical" href="([^"]*)"'
OG_URL = r'<meta property="og:url" content="([^"]*)"'


def test_the_checked_in_template_advertises_no_host_at_all(index_template: str) -> None:
    """The same rule `share_all_template` states for the copied share string.

    "hardcoding a domain here would post the wrong link from every preview
    deploy and from localhost". A checked-in `https://launder.sshh.io` would
    make every Railway preview unfurl as production, so the link in the post
    goes somewhere other than the build being discussed — and it would do it
    silently, because the page still looks right.
    """
    assert _tag_value(index_template, CANONICAL) == "/"
    assert _tag_value(index_template, OG_URL) == "/"


def test_the_rendered_page_names_the_origin_the_request_arrived_on(
    renderer: BootRenderer, level_n: int
) -> None:
    html = renderer.render(level_n, origin="https://launder-pr-12.up.railway.app") or ""
    assert _tag_value(html, CANONICAL) == "https://launder-pr-12.up.railway.app/"
    assert _tag_value(html, OG_URL) == "https://launder-pr-12.up.railway.app/"


def test_no_origin_leaves_the_two_tags_relative(renderer: BootRenderer, level_n: int) -> None:
    """`""` is "do not claim an origin", not "fall back to production".

    A relative URL is wrong for a crawler, which will not resolve `og:url`; it
    is wrong in the harmless direction. Naming production from a host we could
    not identify is wrong in the direction that misdirects readers.
    """
    html = renderer.render(level_n) or ""
    assert _tag_value(html, CANONICAL) == "/"
    assert _tag_value(html, OG_URL) == "/"


def test_the_origin_is_applied_after_the_cache_and_never_baked_into_it(
    renderer: BootRenderer, level_n: int
) -> None:
    """THE CACHE MUST NOT BE KEYED ON `Host`, and must not leak it either.

    `BootRenderer` caches one rendered page per level because the pristine
    reading behind it costs a tokenize plus a score. `Host` is client-supplied
    and can differ on every request, so folding it into the cached HTML leaves
    only bad options: key the cache on it and hand any client a way to force a
    re-score per request, or do not and serve one visitor's hostname to the
    next. The substitution therefore runs on the way out. Assert both halves:
    each render carries its own origin, and nothing else about the page moved.
    """
    a = renderer.render(level_n, origin="https://a.example") or ""
    b = renderer.render(level_n, origin="https://b.example") or ""
    assert "https://b.example" not in a
    assert "https://a.example" not in b
    assert a.replace("https://a.example/", "/") == b.replace("https://b.example/", "/")


@pytest.mark.parametrize(
    "scheme,host,expected",
    [
        ("https", "launder.sshh.io", "https://launder.sshh.io"),
        ("http", "localhost:8000", "http://localhost:8000"),
        ("https", "launder-pr-12.up.railway.app", "https://launder-pr-12.up.railway.app"),
        # Rejected: the value is about to be written into an HTML attribute, and
        # an allow-list that admits only characters which need no escaping is a
        # far shorter argument than "our escaping is correct".
        ("https", "", ""),
        ("https", 'evil.example"><script>alert(1)</script>', ""),
        ("https", "evil.example/path", ""),
        ("https", "evil.example?q=1", ""),
        ("https", "user:pass@evil.example", ""),
        ("https", "two hosts", ""),
        # An IPv6 literal is legitimate and still rejected: it is not reachable
        # from anywhere a share card gets read, and admitting `[` and `]` buys
        # nothing but a wider surface.
        ("https", "[::1]:8000", ""),
        # Not a web scheme.
        ("javascript", "x", ""),
        ("ftp", "example.com", ""),
    ],
)
def test_safe_origin_is_an_allow_list_not_a_parser(scheme: str, host: str, expected: str) -> None:
    assert safe_origin(scheme, host) == expected


def test_a_hostile_host_never_reaches_the_page(renderer: BootRenderer, level_n: int) -> None:
    hostile = 'evil.example"><script>alert(1)</script>'
    html = renderer.render(level_n, origin=safe_origin("https", hostile)) or ""
    assert "evil.example" not in html
    assert _tag_value(html, OG_URL) == "/"


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


#: The two tags `render_origin` rewrites, matched the way IT matches them
#: rather than by literal text. The literal differs between the file the source
#: tree carries and the file vite emits — `web/index.html` carries a
#: `vite-ignore` on the canonical link (without it vite resolves `href="/"` to
#: `web/` and the build dies with EISDIR), and vite strips that attribute on the
#: way out. A test keyed on one spelling would silently stop testing anything
#: the moment it was handed the other.
URL_TAG_PATTERNS = [
    (r'<link rel="canonical" href="[^"]*"[^>]*>', "rel=canonical href"),
    (r'<meta property="og:url" content="[^"]*"[^>]*>', "og:url content"),
]


@pytest.mark.parametrize("pattern,what", URL_TAG_PATTERNS)
def test_a_head_that_lost_its_url_tag_raises_rather_than_shipping_a_relative_og_url(
    index_template: str, pattern: str, what: str
) -> None:
    """`render_origin` asserts, exactly like the other four regions.

    A deleted `og:url` is invisible: the page renders, the game plays, and the
    only symptom is a share card months later that names whatever URL the
    crawler happened to be given. That is precisely the class of drift
    `_sub_once` exists to turn into a boot-time failure.
    """
    stripped, n = re.subn(pattern, "", index_template)
    assert n == 1, f"{pattern} does not describe exactly one tag in web/index.html"
    with pytest.raises(MissingRegion, match=re.escape(what)):
        render_origin(stripped, "https://launder.sshh.io")


@pytest.mark.parametrize("pattern,what", URL_TAG_PATTERNS)
def test_a_duplicated_url_tag_raises_too(index_template: str, pattern: str, what: str) -> None:
    """Two `og:url` tags is a page that names two origins; a crawler picks one."""
    doubled = re.sub(pattern, lambda m: f"{m.group(0)}\n{m.group(0)}", index_template, count=1)
    with pytest.raises(MissingRegion, match=re.escape(what)):
        render_origin(doubled, "https://launder.sshh.io")


def test_the_BUILT_page_still_carries_both_url_tags() -> None:
    """Every other test here reads `web/index.html`; production reads `dist/`.

    That distinction is usually free, because vite copies the page through. It
    is not free for these two tags: `<link href>` is an ASSET REFERENCE to
    vite, which resolved `href="/"` to the `web/` directory and failed the
    build outright with `EISDIR`. The fix is the `vite-ignore` attribute, which
    vite honours by skipping the node and deleting the attribute — so the tag
    that ships is not byte-identical to the tag that was written, and the file
    `BootRenderer` opens in production is the one nothing had asserted on.
    """
    built = repo_root() / "web" / "dist" / "index.html"
    if not built.is_file():
        pytest.skip("web/dist/index.html is not present in this checkout")
    html = built.read_text(encoding="utf-8")
    for pattern, what in URL_TAG_PATTERNS:
        assert len(re.findall(pattern, html)) == 1, what
    # Comments stripped first: the head's own comment EXPLAINS `vite-ignore`,
    # and an assertion that failed on the paragraph documenting the attribute
    # would be fixed by deleting the documentation.
    markup = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    assert "vite-ignore" not in markup, "vite left its own opt-out attribute in the shipped page"
    # ...and `render_origin` still matches against the BUILT text, not just the
    # source it was written against.
    rendered = render_origin(html, "https://launder.sshh.io")
    assert rendered.count('"https://launder.sshh.io/"') == 2


def test_rendering_leaves_the_rest_of_the_head_alone(renderer: BootRenderer, level_n: int) -> None:
    """The share card and the browser chrome survive the five substitutions.

    These strings are static document metadata rather than copy.toml keys (the
    reasoning is written out in web/index.html's head), which means nothing
    else in the pipeline is watching them. If a future region rewrite ate the
    description or a theme-color, the first report would be a bad-looking
    unfurl on Hacker News.
    """
    html = renderer.render(level_n) or ""
    assert re.search(r'<meta\s+name="description"\s+content="[^"]+"', html) is not None
    assert '<meta property="og:title" content="Launder WM" />' in html
    assert '<meta property="og:site_name" content="Launder WM" />' in html
    assert '<meta name="twitter:card" content="summary" />' in html
    assert html.count('<meta name="theme-color"') == 2
    assert 'rel="icon"' in html


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
async def test_the_share_card_names_the_host_the_visitor_actually_used(
    page_client: httpx.AsyncClient,
) -> None:
    """Preview deploys unfurl as themselves, localhost as localhost."""
    r = await page_client.get("/", headers={"Host": "launder-pr-12.up.railway.app"})
    assert _tag_value(r.text, OG_URL) == "http://launder-pr-12.up.railway.app/"
    assert _tag_value(r.text, CANONICAL) == "http://launder-pr-12.up.railway.app/"


@pytest.mark.anyio
async def test_a_forged_host_header_is_not_reflected_into_the_page(
    page_client: httpx.AsyncClient,
) -> None:
    """`Host` is client-supplied, and this is the one place it is written back
    into the document. `safe_origin` rejects it and the tags stay relative."""
    r = await page_client.get("/", headers={"Host": 'evil.example"><script>x</script>'})
    assert r.status_code == 200
    assert "evil.example" not in r.text
    assert _tag_value(r.text, OG_URL) == "/"


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
