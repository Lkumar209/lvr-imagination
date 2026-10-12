# Would Imagination Help? (working title)

Paper for the **Latent Visual Reasoning workshop @ WACV 2027**, non-archival track
(4 pages + references; arXiv after acceptance). Deadline **Oct 18, 2026, 07:00 EDT** via OpenReview.

**Question.** VLMs reason about spatial and physical changes ("what if I walk
around?", "what if these swap?", "will this fall?") in text. Suppose the model
could *imagine* the changed scene perfectly. How much would that help? How much
of that gain do real generative models recover? And does a *wrong* imagined
image hurt, which would show the model actually uses the visual state?

**Setup.** Synthetic scenes where the imagined state can be rendered exactly.
Each question has three renders:

| state | meaning |
|---|---|
| `original` | what the model sees |
| `imagined` | oracle: exactly the state the question asks it to picture |
| `wrong` | plausible but incorrect imagined state with a *different* answer |

Conditions (`scripts/evaluate.py`): `direct` (image only), `cot` (describe the
changed scene in words, then answer), `oracle` (+ imagined), `oracle_only`
(imagined alone), `wrong` (+ misleading imagined), `shuffled` (+ another
question's imagined image), `recon` (+ the model's *own* imagination: it
writes the scene down as numbers, we apply the change and render it;
`scripts/recon.py`). All two-image conditions share one prompt wording.

## Tasks

| task | asks | imagined state |
|---|---|---|
| `viewpoint` | after walking a quarter or half way around, is A left or right of B? | render from the new viewpoint |
| `swap` | after A and B trade places, which object is closest to A? | render of the rearranged scene |
| `stability` | will this stack of blocks stay standing or fall? | the aftermath (fallen blocks on the floor, or unchanged) |

Answers come from the geometry, with margins so no question sits on a
borderline (screen gap ≥ 8% of width; nearest/second-nearest distance ratio
≥ 1.4; centre of mass ≥ 0.08 m from a tipping edge). Binary answers are balanced
by seed. For `viewpoint`, half the questions have an answer that differs from
the current view, so "answer from the current view" scores 50%. For `swap`, the
answer is never the swap partner nor the target's nearest neighbour before the
swap (an earlier version always had a swapped object as the answer, a shortcut).

## Code

| path | purpose |
|---|---|
| `imagine/render.py` | numpy ray tracer (spheres, boxes, cylinders, floor, shadows) + object-id buffer |
| `imagine/tasks.py` | the three question generators |
| `scripts/generate.py` | builds `questions.jsonl` + renders, in parallel |
| `scripts/contact_sheet.py` | original / imagined / wrong strips for eyeballing |
| `scripts/evaluate.py` | run a VLM under each condition (`hf`, `openai`, or `mock` backend); resumable |
| `scripts/recon.py`, `imagine/recon.py` | model-built imagination for the `recon` condition |
| `scripts/score.py` | accuracy with bootstrap CIs, follow-the-image rates, diagnostic splits |
| `tests/` | geometry and answer checks (`python -m pytest tests`) |
| `paper/` | WACV 2027 author kit |

```bash
python scripts/generate.py --n 8 --out data/pilot
python scripts/contact_sheet.py data/pilot --task stability
```

Pure numpy + Pillow; no GPU needed for generation. On GPU2 (48 CPU workers)
the 600-question `data/v1` set takes a few minutes. Full model runs:
`scripts/run_model.sh <gpu> <hf model id>` (run from the repo root) on GPU2 (vlmjudge env, 4-bit, cached weights
in `/shared/scratch/0/home/v_laxya_kumar/hf_cache`).

## Known simplifications

- Stability is judged on the centre of mass along x only (offsets are only
  along x, perpendicular to the camera). The aftermath image is a plausible
  depiction (blocks lying on the overhang side), not a physics simulation.
- For stable stacks the imagined image equals the original, which is what perfect
  imagination of "nothing happens" looks like. Keep this in mind when reading
  oracle gains on `stability`.
