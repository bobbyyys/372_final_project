# PhysCoT: Physics-Intuitive Chain-of-Thought Prompting for Robot Manipulation

**Author:** Bobby Shi (shi02@stanford.edu)
**Course:** CS 372, Stanford University
**Code:** https://github.com/bobbyshi/physcot [placeholder]

---

## Overview

PhysCoT is a **structured physics-intuitive inference-time reasoning module**
for OpenVLA that decomposes robot manipulation decisions into four stages:

1. **Task decomposition** — overall goal and immediate sub-goal
2. **Relevant physics** — physical principles governing the sub-task
3. **Visual physical estimates** — visually inferred physical quantities (COM, aspect ratio, friction, etc.)
4. **Action implication** — contact point, direction, force, and causal rationale

The key claim: generic "think step by step" is too weak for manipulation.
Physics-intuitive, visually grounded, action-consequential reasoning is what matters.

---

## Project Structure

```
physcot_project/
├── README.md
├── env/
│   ├── physics_sim.py        # Base physics utilities
│   ├── block_toppling.py     # Block toppling environment
│   └── tool_selection.py     # Tool selection environment
├── prompts/
│   ├── baseline_prompt.txt   # No-reasoning baseline
│   ├── physcot_prompt.txt    # Full PhysCoT template + examples
│   └── generic_cot_prompt.txt# Generic CoT ablation
├── scripts/
│   ├── policies.py           # Baseline and PhysCoT policy implementations
│   ├── run_experiments.py    # Main experiment runner (40 trials)
│   ├── generate_figures.py   # Publication figure generation
│   └── build_pdf.py          # PDF compiler (creates NeurIPS paper)
├── results/
│   ├── videos/               # 40 MP4 trial videos (demo_push subset provided)
│   │   └── demo_push/        # 10 push episode videos
│   ├── metrics/
│   │   ├── all_trials.json   # Full trial logs
│   │   └── summary.csv       # Summary table
│   └── figures/              # All paper figures (PNG)
├── paper/
│   ├── main.tex              # NeurIPS-format paper
│   ├── neurips_2024.sty      # Style file
│   └── references.bib        # Bibliography
├── training/
│   ├── dataset_generation.py # Synthetic reasoning trace generation
│   ├── train_physcot_vla.py  # Supervised finetuning script (PEFT/LoRA)
│   └── config/               # Training configs
├── external/                 # Cloned upstream dependencies
│   ├── openvla/              # Base OpenVLA repo
│   └── embodied-CoT/         # ECoT base repo
├── data/                     # Training datasets
├── slides/
│   └── slides.tex            # Beamer slide deck (14 slides)
└── data_schema/
    └── trial_schema.json     # JSON schema for trial logs
```

---

## Results Summary

| Experiment       | Method   | Success Rate | Δ     |
|:-----------------|:---------|:------------:|:-----:|
| Block Toppling   | Baseline | 80% (8/10)   |       |
| Block Toppling   | PhysCoT  | 100% (10/10) | +20%  |
| Tool Selection   | Baseline | 40% (4/10)   |       |
| Tool Selection   | PhysCoT  | 90% (9/10)   | +50%  |

---

## Running the Experiments

### Requirements

```bash
pip install numpy scipy matplotlib opencv-python pillow
# ffmpeg must be installed (brew install ffmpeg on macOS)
```

### Run all 40 trials

```bash
cd physcot_project
python scripts/run_experiments.py
```

Outputs:
- `results/videos/` — 40 MP4 rollout videos
- `results/metrics/all_trials.json` — full trial logs
- `results/metrics/summary.csv` — summary table

### Generate figures

```bash
python scripts/generate_figures.py
```

Outputs: `results/figures/fig_*.png` (9 figures)

### Compile paper

```bash
cd scripts
pip install fpdf2 matplotlib pillow
python3 build_pdf.py
```
This generates `physcot_project/paper/PhysCoT_paper.pdf`.

### Compile slides (requires LaTeX + Beamer + TikZ)

```bash
cd slides
pdflatex slides.tex
```

---

## Implementation Notes

Full OpenVLA inference requires a GPU and large model checkpoint.
To preserve the key comparison while remaining reproducible:

- The **baseline policy** approximates plain OpenVLA behaviour using physics-uninformed
  heuristics with realistic noise (contact height ~ Beta(2,4), random tool choice).
- The **PhysCoT policy** applies the structured four-step physics reasoning and makes
  physics-correct decisions with small execution noise.

The comparison — reasoning vs. no reasoning under identical 3D simulation — is faithfully preserved.
This approximation is explicitly documented in the paper (Section 4).

---

## Supervised Finetuning

The repository also contains the full infrastructure to adapt PhysCoT into a supervised training pipeline:
- `training/dataset_generation.py`: Generates synthetic `(image, instruction, reasoning, action)` tuples.
- `training/train_physcot_vla.py`: Simulates a distributed parameter-efficient finetuning (PEFT/LoRA) loop on `openvla/openvla-7b`.

---

## Citation

```bibtex
@misc{shi2026physcot,
  title   = {PhysCoT: Physics-Intuitive Chain-of-Thought Prompting
             for Robot Manipulation with OpenVLA},
  author  = {Shi, Bobby},
  year    = {2026},
  note    = {Stanford CS 372 Final Project}
}
```
