# VLM Categorization — Levering (2020) Real-Time Simulation

This project simulates the Levering (2020) categorization task in real time, treating LLMs and VLMs as active participants. Models receive trial-by-trial feedback and learn the category rule from scratch, mirroring the original human experiment.

## Task

Stimuli are 3-digit codes (`XYZ`) encoding three binary visual dimensions:

| Digit | Dimension | 1 | 2 |
|-------|-----------|---|---|
| X | Shape | Square | Triangle |
| Y | Size | Large (1.50 in) | Small (0.75 in) |
| Z | Shading | Black | White |

**Two conditions** from Levering (2020), sourced from the [Psych-101 dataset](https://huggingface.co/datasets/marcelbinz/Psych-101):

| Condition | Labels | Rule | Exception items |
|-----------|--------|------|-----------------|
| **NLS** (exp1, 126 participants) | Z / W | Non-linearly separable — no single feature predicts category | `212`, `121` contradict the shape heuristic |
| **LS** (exp2, 102 participants) | R / H | Linearly separable — majority-of-digits rule (≥2 ones → R, ≥2 twos → H) | None |

Each condition uses 6 training stimuli and 8 test stimuli.

---

## Notebook: `simulation_final.ipynb`

### Pipeline 1 — Text Simulation (LLMs)

Models receive stimuli as numeric codes (e.g., `211`) accumulated in their context window, with the full trial history prepended to each new trial. After each categorization response, they receive correct/incorrect feedback and a point score.

**Training:** `N_REPS` shuffled blocks over the 6 training stimuli per condition.

**Test block:** Models categorize all 8 stimuli (including 2 novel ones not seen in training) and provide a typicality rating (1–9 scale) for each.

### Pipeline 2 — Visual Simulation (VLMs)

Stimuli are rendered as PNG images (square or triangle, large or small, black or white on a gray background) and sent via the multimodal API. The full trial history is passed as text, but each new trial's stimulus is image-only — no numeric code shown to the model.

A multi-turn message structure preserves the full image history so the model sees each stimulus in sequence, as human participants do.

---

## Analyses & Figures

| Figure | Description |
|--------|-------------|
| **Fig 1** | Block-level learning curves — text pipeline vs. human benchmarks (Levering 2020, Fig. 4) |
| **Fig 2** | Three-panel summary: (a) NLS vs. LS accuracy gap, (b) exception vs. non-exception accuracy, (c) token bias diagnostic (standard vs. counterbalance-corrected) |
| **Fig 3** | Learning curves — visual pipeline vs. human benchmarks |
| **Fig 4** | Visual pipeline: train vs. test accuracy, typicality ratings per stimulus |
| **Fig 5/6** | Text vs. visual pipeline comparison — learning curves and exception item breakdown |
| **Typicality** | Typicality ratings per stimulus: human average (Psych-101) vs. text pipeline vs. visual pipeline |

### Key design choices

**Counterbalancing:** Each run is averaged across the standard label assignment and a swapped version (e.g., Z↔W) to cancel token-level response bias toward any single label.

**Relabeling diagnostic (Llama NLS only):** Two relabeled variants are run — `label_only` (Z→P, W→Q) and `surface_relabel` (Z→P, W→Q + digit-position permutation XYZ→ZYX) — to test whether learning transfers under surface-level changes.

**Seeds:** Results are averaged across seeds `[2, 5, 3]` with standard error bars.

---

## Models

Accessed via Brown CCV's LiteLLM proxy (`https://litellm.ccv.brown.edu`).

| Model | Pipeline |
|-------|----------|
| `gpt-5.2` | Text + Visual |
| `Llama-3.3-70B-Instruct` | Text |
| `claude-sonnet-4-5` | Text + Visual |
| `gemini-2.5-pro` | Text + Visual |
| `gemini-3-flash-preview` | Text + Visual |

---

## Setup

1. Install dependencies:
```bash
pip install openai python-dotenv datasets tqdm numpy pandas matplotlib
```

2. Create a `.env` file with your API key:
```
OPENAI_API_KEY=your_key_here
```

3. Run `simulation_final.ipynb`. Set `QUICK_TEST = False` for a full production run (`N_REPS = 6`, all models, both pipelines). Set `QUICK_TEST = True` for a fast smoke test (`N_REPS = 3`, first model only, no visual pipeline).

---

## References

- Levering, K. R. (2020). *Revisiting the role of typicality in category learning.* Journal of Experimental Psychology: Learning, Memory, and Cognition.
- Binz, M. et al. (2025). *Centaur: a foundation model of human cognition.*
- Psych-101 dataset: [marcelbinz/Psych-101](https://huggingface.co/datasets/marcelbinz/Psych-101)
