# Would Imagination Help?

An oracle study of visual intermediate states for spatial and physical reasoning
in VLMs. Paper for the **Latent Visual Reasoning workshop @ WACV 2027**
(non-archival, 4 pages + references; arXiv after acceptance). Deadline
**Oct 18, 2026, 07:00 EDT** via OpenReview. Draft: `paper/main.pdf`.

**Question.** VLMs reason about change ("what if I walk around?", "what if these
swap?", "will this fall?") in text. If a model could *see* the changed scene
perfectly, how much would it help? Does it actually use that image? How faithful
must the image be, and how close do models get when they build it themselves?

**Setup.** Synthetic scenes where the imagined state can be rendered exactly.
The model, question and prompt stay fixed; only the second image changes.

| condition (paper / code) | second input |
|---|---|
| Img. / `direct` | none |
| Text / `cot` | none; the model first describes the change in words |
| Own / `recon` | the model's own reconstruction, transformed and rendered (`scripts/recon.py`) |
| Oracle / `oracle` | the exact imagined state |
| Unrel. / `shuffled` | another question's imagined state |
| Misl. / `wrong` | a plausible imagined state with a different answer |
| — / `oracle_only` | the imagined state *instead of* the original |

All two-image conditions share one prompt wording.

## Tasks

| task | asks | imagined state |
|---|---|---|
| `viewpoint` | after walking a quarter or half way around, is A left or right of B? | render from the new viewpoint |
| `swap` | after A and B trade places, which object is closest to A? | render of the rearranged scene |
| `stability` | will this stack of blocks stay standing or fall? | the aftermath (fallen blocks on the floor, or unchanged) |

200 questions per task. Answers come from the geometry, with margins so no
question is borderline (screen gap ≥ 8% of width; nearest/second-nearest distance
ratio ≥ 1.4; centre of mass ≥ 0.08 m from a tipping edge). Binary answers are
balanced by seed. For `viewpoint`, half the answers differ from the current view.
For `swap`, the answer is never the swap partner nor the target's nearest
neighbour before the swap, and the partner is not listed as an option (an earlier
version had both shortcuts; see the paper).

## Reproducing the paper

Everything is deterministic (seeded generation, greedy decoding). Generation needs
only numpy + Pillow; model runs need a GPU with `transformers>=4.57`,
`bitsandbytes`, `accelerate` (we used one 24 GB A5000 per model, 4-bit weights).

```bash
# 1. data: 600 questions with original / imagined / wrong renders
python scripts/generate.py --n 200 --out data/v1 --workers 48

# 2. per model (Qwen/Qwen3-VL-8B-Instruct, Qwen/Qwen2.5-VL-7B-Instruct, OpenGVLab/InternVL3-8B-hf with --quant 4;
#    Qwen/Qwen2.5-VL-3B-Instruct without --quant)
M=Qwen/Qwen3-VL-8B-Instruct; TAG=Qwen3-VL-8B-Instruct-q4
python scripts/evaluate.py data/v1 --backend hf --model $M --quant 4 \
    --conditions direct oracle oracle_only wrong shuffled
python scripts/recon.py data/v1 --backend hf --model $M --quant 4
python scripts/evaluate.py data/v1 --backend hf --model $M --quant 4 --conditions recon --recon $TAG
python scripts/evaluate.py data/v1 --backend hf --model $M --quant 4 --conditions cot --limit 100 --max-tokens 2048
#    (Qwen3-VL-8B swap CoT on all 200: add --tasks swap without --limit)

# 3. degraded oracles and rescaled reconstructions (Qwen models)
python scripts/degrade.py data/v1 --sigmas 0 0.1 0.25 0.5 1.0
for s in 0 0.1 0.25 0.5 1; do
  python scripts/evaluate.py data/v1 --backend hf --model $M --quant 4 --conditions recon \
      --recon noise_$s --out results_noise/s$s
done
python scripts/recon.py data/v1 --rescale-from $TAG
python scripts/evaluate.py data/v1 --backend hf --model $M --quant 4 --conditions recon \
    --recon $TAG-rescaled --out results_rescaled

# 4. prompt robustness (Qwen3-VL-8B, 50 per task)
python scripts/evaluate.py data/v1 --backend hf --model $M --quant 4 \
    --conditions direct oracle wrong --limit 50 --prompt-variant alt

# 5. stability controls: second images from a camera turned 20 degrees, and pixel-change checks
python scripts/camera_turn.py data/v1 data/v1_camturn --deg 20
python scripts/evaluate.py data/v1_camturn --backend hf --model $M --quant 4 --conditions oracle wrong --out results_camturn
python scripts/extra_checks.py           # -> results/extra_checks.json (read by audit_numbers.py)

# 6. tables, figures, statistics, audit
python scripts/make_table.py results/v1 paper/tables/main.tex \
    --models Qwen3-VL-8B-Instruct-q4 Qwen2.5-VL-7B-Instruct-q4 InternVL3-8B-hf-q4
python scripts/make_supp_tables.py paper/tables/supp.tex
python scripts/significance.py --out paper/tables/significance.txt
python scripts/fig_noise.py data/v1 paper/figs/noise.pdf --models Qwen3-VL-8B-Instruct-q4 Qwen2.5-VL-7B-Instruct-q4
python scripts/audit_numbers.py          # every number in the prose vs. the results
```

`scripts/run_model.sh` is the GPU2 wrapper we used for step 2 (cluster paths
inside; adjust before use). Results are committed under `results*/` (small
JSONL); `data/` is not, since step 1 regenerates it exactly.

## Code

| path | purpose |
|---|---|
| `imagine/render.py` | numpy ray tracer (spheres, boxes, cylinders, floor, shadows) + object-id buffer |
| `imagine/tasks.py` | the three question generators |
| `imagine/recon.py` | turn a model's JSON scene description into a rendered imagined state |
| `scripts/generate.py` | builds `questions.jsonl` + renders, in parallel |
| `scripts/evaluate.py` | run a VLM under each condition (`hf`, `openai`, or `mock` backend); resumable |
| `scripts/recon.py` | the first stage of the `recon` condition, and `--rescale-from` |
| `scripts/degrade.py` | oracles with Gaussian position noise |
| `scripts/score.py` | accuracy with bootstrap CIs, follow-the-image rates, diagnostic splits |
| `scripts/analyze_imagined.py` | perception vs. reading errors on an imagined second image |
| `scripts/camera_turn.py` | stability control: oracle and misleading images from a turned camera |
| `scripts/extra_checks.py` | pixel change of stability second images; keyword count on Text traces |
| `scripts/significance.py` | exact McNemar tests, Holm-corrected |
| `scripts/make_table.py`, `scripts/make_supp_tables.py` | LaTeX tables from the results |
| `scripts/fig_*.py` | paper figures |
| `scripts/audit_numbers.py` | checks the numbers in the paper's prose against the results |
| `scripts/contact_sheet.py` | original / imagined / wrong strips for eyeballing |
| `tests/` | geometry and answer checks (`python -m pytest tests`) |
| `paper/` | the paper (WACV 2027 author kit) |

## Known simplifications

- Stability is judged on the centre of mass along x only (offsets are only
  along x, perpendicular to the camera). The aftermath image is a plausible
  depiction (blocks lying on the overhang side), not a physics simulation.
- For stable stacks the imagined image equals the original. Given the aftermath
  alone, models answer with a fixed prior, so `oracle_only` is not interpreted
  on `stability`.
- Main models run with 4-bit NF4 weights.
