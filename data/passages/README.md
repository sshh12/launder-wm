# data/passages/

`forge pack` and `forge publish` write here: `<id>.public.json` (shipped),
`<id>.server.json` (claims + par, shipped) and `<id>.author.json` (**the answer
key — repo only, never served, excluded from the image by `.dockerignore` and
refused by the `/data/passages` mount**).

**This directory is EMPTY in the committed tree, and that is a real gap, not an
oversight.** A passage has to be *generated with the watermark on*, which needs
all of:

1. a CUDA torch build on an RTX-class GPU (`uv sync --extra cuda`; check with
   `uv run forge doctor`),
2. accepted Gemma Terms of Use and an `HF_TOKEN` for `google/gemma-3-4b-it`
   (the weights are gated),
3. `forge gen -> analyze -> triage -> tune -> solve -> claims`,
4. `forge pack <id> --level L2 --in <candidates.jsonl> --candidate <cid>`, then
   `forge publish --date <YYYY-MM-DD> --passage-id <id>`.

Until then `launder-serve` serves the DEV FIXTURE (`data/dev/passage.txt`, id
`p_dev`, `dev: true` in the boot payload) so the page is playable in
development. That fixture is HUMAN PROSE reading z = 1.31 against a notch of
2.3263 — it starts *below the line*, so it is not a puzzle, which is exactly why
it is never scheduled and is refused in production.

`.gitkeep` is load-bearing: git cannot commit an empty directory, and the
`Dockerfile`'s `COPY data/passages/ /data/passages/` fails the build on a fresh
clone if the path does not exist.
