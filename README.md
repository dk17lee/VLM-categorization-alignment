# VLM Categorization — Levering (2020) Replication

This project tests whether LLMs and VLMs behave like humans on the Levering (2020) categorization task, using two complementary approaches: **real-time simulation** and **alignment scoring against real human data**.

## Task

Stimuli are 3-digit codes (`XYZ`) encoding shape, size, and shading:
- `X` — shape: `1` = square, `2` = triangle
- `Y` — size: `1` = large (1.50 in), `2` = small (0.75 in)
- `Z` — shading: `1` = black, `2` = white

**Two conditions** from Levering (2020), pulled from the [Psych-101 dataset](https://huggingface.co/datasets/marcelbinz/Psych-101):

| Condition | File | Categories | Rule |
|-----------|------|------------|------|
| NLS | `exp1.csv` (126 participants) | Z / W | Non-linearly separable — no single feature predicts category; exception items `212`, `121` contradict the shape heuristic |
| LS | `exp2.csv` (102 participants) | R / H | Linearly separable — majority-of-digits rule (≥2 ones → R, ≥2 twos → H) |

---

## Notebooks

### `simulation_final.ipynb` — Real-Time Simulation

Models are treated as active participants. They receive feedback after each trial and accumulate the full trial history in their context, mirroring the original human experiment.

**Two pipelines:**
- **Text pipeline (LLMs):** stimuli presented as numeric codes (e.g., `211`)
- **Visual pipeline (VLMs):** stimuli rendered as PNG images (grey background, black/white shapes)

**Training:** `N_REPS` shuffled blocks over the 6 training stimuli per condition, with correct/incorrect feedback.

**Test block:** models categorize all 8 stimuli and provide typicality ratings (1–9 scale).

**Analyses:**
- Block-level learning curves vs. human benchmarks (Levering et al. 2020, Figure 4)
- Exception vs. non-exception item accuracy (NLS only; exception items `212`, `121` contradict shape rule)
- Typicality ratings: model vs. human averages from Psych-101
- Counterbalancing over swapped/standard label maps to cancel token-level response bias
- Surface relabeling diagnostic (Llama NLS only) to check for memorization

---

### `categorization_alignment_final.ipynb` — Human Alignment Scoring

Rather than simulating, this notebook replays **real human transcripts** and measures how surprised a model is by the human's actual choices using **negative log-likelihood (NLL)**.

Lower NLL = model assigns higher probability to what the human did = better alignment.

**Three analyses:**

1. **Condition comparison** — NLL per model on NLS vs. LS, training and test phases, with SE error bars across participants

2. **Contamination / memorization check** — standard prompts (W/N labels) vs. relabeled prompts (W→P, N→Q, plus stimulus code permutation `XYZ→ZYX`). A large NLL increase under relabeling suggests the model was leveraging memorized associations rather than genuinely learning the rule.

3. **VLM input variants** — three ways to present stimuli to vision models:
   - `text_only` — numeric code in plain text
   - `image_only` — rendered PNG replaces the stimulus code
   - `image_and_text` — PNG plus code caption

**Also computed:** Spearman correlation between model confidence (P(human's category)) and human typicality ratings on test trials.

---

## Models

Accessed via Brown CCV's LiteLLM proxy (`https://litellm.ccv.brown.edu`).

| Model | Type | Used in |
|-------|------|---------|
| `gpt-5.2` | LLM + VLM | Simulation |
| `Llama-3.3-70B-Instruct` | LLM | Simulation, Alignment |
| `claude-sonnet-4-5` | VLM | Alignment |
| `gpt-5.4` | VLM | Alignment |

---

## Setup

1. Clone the repo and install dependencies:
```bash
pip install openai python-dotenv datasets tqdm numpy pandas matplotlib scipy
```

2. Create a `.env` file with your API key:
```
OPENAI_API_KEY=your_key_here
```

3. Run `simulation_final.ipynb` for the real-time simulation, or `categorization_alignment_final.ipynb` for the alignment analysis.

---

## References

- Levering, K. R. (2020). *Revisiting the role of typicality in category learning.* Journal of Experimental Psychology: Learning, Memory, and Cognition.
- Binz, M. et al. (2025). *Centaur: a foundation model of human cognition.*
- Psych-101 dataset: [marcelbinz/Psych-101](https://huggingface.co/datasets/marcelbinz/Psych-101)
