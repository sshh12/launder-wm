# data/passages/

The fifteen campaign passages, `p01` .. `p15`. Each one is three files:
`<id>.public.json` (shipped), `<id>.server.json` (claims + par, shipped) and
`<id>.author.json` (**the answer key — repo only, never served, excluded from
the image by `.dockerignore` and refused by the `/data/passages` mount**).

**The id says nothing about where a passage sits in the campaign.**
`data/config/progression.toml` maps level number -> passage id -> ruleset, and
it is the only place that ordering lives. The ids are numbered in campaign
order today because that is easier to read, not because anything derives one
from the other — renumbering the campaign is an edit to `progression.toml`
alone.

A passage has to be *generated with the watermark on*, which needs all of:

1. a CUDA torch build on an RTX-class GPU (`uv sync --extra cuda`; check with
   `uv run forge doctor`),
2. accepted Gemma Terms of Use and an `HF_TOKEN` for `google/gemma-3-4b-it`
   (the weights are gated),
3. `forge gen -> analyze -> triage -> tune -> solve -> claims`,
4. `forge pack <id> --level L2 --in <candidates.jsonl> --candidate <cid>`,
   which writes the three files above. Adding a `[[level]]` block naming that
   id in `data/config/progression.toml` is what actually puts it in the game.

`launder-serve` serves the DEV FIXTURE (`data/dev/passage.txt`, id `p_dev`,
`dev: true` in the boot payload) only when this directory is empty, so the page
is still playable on a machine that cannot generate one. That fixture is HUMAN
PROSE reading z = 1.31 against a notch of 2.3263 — it starts *below the line*,
so it is not a puzzle, which is exactly why it is never part of the campaign
and is refused in production.

`.gitkeep` is load-bearing: git cannot commit an empty directory, and the
`Dockerfile`'s `COPY data/passages/ /data/passages/` fails the build on a fresh
clone if the path does not exist.
