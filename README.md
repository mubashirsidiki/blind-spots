# Urdu Nastaliq Blind Spot in Vision-Language Models

**Fatima Fellowship Technical Challenge: Blind Spots of Frontier Models**

Systematic evaluation of [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) on bilingual Pakistani restaurant menus containing side-by-side English (Latin) and Urdu (Nastaliq) text.

---

## 1. Lived Experience and the Blind Spot (Question 1)

Urdu is spoken by over 230 million people worldwide and is the national language of Pakistan. In Pakistan, Urdu is almost never printed in standard Arabic Naskh. From street signboards, restaurant menus, and store receipts to national newspapers and school textbooks, Urdu print culture is written exclusively in the **Nastaliq** calligraphic style.

Nastaliq typography presents distinct visual challenges that standard Arabic OCR engines do not encounter:
- **Cascading Slanted Baselines**: Words form along diagonal slants (roughly 30° to 45°), moving from upper right to lower left rather than along a flat horizontal line.
- **Stacked Vertical Ligatures**: Characters stack on top of each other in multiple vertical tiers within a single word cluster.
- **Contextual Glyph Variants and Expanded Alphabet**: Letter shapes change based on their vertical tier, and Urdu adds retroflex and aspirated sounds not present in Arabic.

Multimodal vision-language benchmarks (such as DocVQA, OCRBench, and CC-OCR-Bench) report strong Arabic script scores, but these evaluations rely almost exclusively on horizontal Naskh typography. When modern open-weight frontier models encounter authentic Pakistani documents in Nastaliq, their OCR capability breaks down completely.

---

## 2. Systematic Evaluation on Qwen3-VL-2B-Instruct (Question 2)

We evaluated [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) (2 billion parameters, bfloat16) on authentic menu pages from Noorani Kabab House in Karachi, Pakistan. The dataset contains 60 manually annotated items (58 core restaurant dishes with numerical prices and portion sizes across pages 2 to 5, plus 2 cover notice items on page 1).

### Quantitative Benchmark Results

Evaluated on 58 core restaurant dishes (pages 2 to 5):

| Metric | English (Latin) | Urdu (Nastaliq) | Disparity |
|---|---|---|---|
| **Exact Match Accuracy** | 98.3% (57/58) | 1.7% (1/58) | -96.6% drop |
| **Character Error Rate (CER)** | 0.0010 | 0.5901 | 590.1x higher |
| **Word Error Rate (WER)** | 0.0057 | 0.9368 | 164.4x higher |
| **Price Accuracy** | 100.0% (58/58) | N/A | Full coverage |

Across the full 60-item dataset (including page 1 notices), English exact match is 98.3% (CER 0.0010) and Urdu exact match is 1.7% (CER 0.5825), maintaining a 582.5x error disparity.

### Primary Failure Modes

1. **Ligature Fracturing**: Cartesian patch tokenization (16×16 pixel patches in ViT) splits diagonal, vertically stacked character clusters across disconnected visual tokens. The model misclassifies stacked ligatures into unrelated words (e.g. *ہماری* → *جاری*, *برانچ* → *برائے*).
2. **Phonetic Back-Transliteration**: When visual features from the Urdu column carry high entropy, the decoder defaults to its dominant language prior. It reads the adjacent English column and spells the English word phonetically in Arabic script (e.g. reading "(WITH BUTTER)" and outputting *(بٹر)* instead of reading the Urdu text *(مکھن کے ساتھ)*).
3. **Consonant and Dot Substitutions**: Misidentifying Urdu retroflex characters and consonant clusters (e.g. *چپاتی* → *شاپاتی*).

---

## 3. Proposed Path Forward (Question 3)

Closing this capability gap requires adjustments across data curation, architecture, and fine-tuning:

1. **Data Curation Strategy (Paired Synthetic Nastaliq Corpus)**:
   Pretraining datasets contain billions of Latin and standard Naskh web crawls, but virtually zero paired Nastaliq corpora. Using modern Nastaliq typography rendering engines (such as Jameel Noori Nastaleeq and Mehr Nastaliq), synthetically generate diverse, high-resolution document pages with varying slants, tiered ligatures, lighting conditions, and camera angles. Fusing this data into multimodal pretraining would establish baseline visual feature representations for Nastaliq glyphs.

2. **Architectural Adjustment (Curvilinear and Diagonal-Aware Tokenization)**:
   Current Vision Transformers enforce rigid Cartesian 2D grid embeddings (height and width). Extending spatial position encoding (such as Interleaved-MRoPE) to support non-Cartesian, diagonal scanning trajectories would preserve semantic continuity across cascading vertical tiers without fracturing stacked ligatures.

3. **Fine-Tuning Paradigm (Cross-Script Attention Regularization)**:
   In bilingual hybrid layouts (English left, Urdu right), the decoder attends heavily to the Latin column when visual confidence in the vernacular script is low. Introducing an attention penalty or spatial masking loss during supervised fine-tuning would prevent cross-column visual leakage, forcing the language model to decode directly from the Urdu visual tokens rather than back-transliterating phonetic English.

---

## Repository Structure

```
├── dataset/
│   └── menu/
│       ├── 1.jpg - 5.jpg           # High-resolution menu page photographs
│       └── ground_truth.json       # 60 annotated items with English, Urdu, and prices
├── scripts/
│   ├── evaluate_qwen3_vl.py       # Batch VLM inference with GPU latency tracking
│   └── benchmark_metrics.py       # Levenshtein distance, DP alignment, and price scoring
├── results/
│   ├── qwen3_vl_eval_results.json # Raw model generation outputs and timings
│   ├── benchmark_scores.json      # Complete quantitative benchmark metrics
│   └── evaluation_summary.md      # Human-readable markdown inference log
├── concepts/                       # Diagram assets illustrating Naskh vs. Nastaliq
└── urdu_nastaliq_blindspot.ipynb   # Interactive analysis and live inference notebook
```

## Quickstart

### Prerequisites

- Python 3.10+
- PyTorch with CUDA support (tested on NVIDIA RTX 4060 with 8GB VRAM)
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
git clone https://github.com/mubashirsidiki/blind-spots.git
cd blind-spots
uv sync
```

### Running the Notebook

Open and run `urdu_nastaliq_blindspot.ipynb` in your Jupyter environment. The notebook loads `Qwen3-VL-2B-Instruct` locally or from Hugging Face, executes interactive inference on a sample menu page, and verifies quantitative benchmark metrics live.

### Reproducing Benchmark Scores

To re-run metric alignment and scoring:

```bash
uv run python scripts/benchmark_metrics.py
```

To re-run full batch inference across all 5 pages:

```bash
uv run python scripts/evaluate_qwen3_vl.py
```

## References

- Qwen Team (2025). [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388). arXiv:2505.09388 [cs.CL].
- Model Card: [Qwen/Qwen3-VL-2B-Instruct on Hugging Face](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct).
