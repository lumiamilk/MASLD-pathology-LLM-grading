# The Extraction-Reasoning Gap: A Dual-Condition Benchmark of 28 Open-Source LLMs for Grading Liver Pathology Reports

[![DOI](https://zenodo.org/badge/1207830208.svg)](https://doi.org/10.5281/zenodo.20162414)

This repository contains the official code and sample data for the paper submitted to *Scientific Reports* (Submission ID: 114417b3-3dfc-47ab-b6a1-66bce79fe493).

## Abstract

We evaluate 28 open-source large language models (LLMs, 4B--32B parameters) on a dual-condition benchmark for grading unstructured liver pathology reports in Chinese. The benchmark comprises two complementary tasks: **Condition A** (extraction from complete reports with explicit diagnostic conclusions) and **Condition B** (reasoning from microscopy descriptions without diagnostic conclusions). We quantify a systematic "inference gap" of ~0.30 Macro F1 across all models when transitioning from extraction to reasoning, demonstrate a +0.33 F1 advantage for Chinese-native models over English-native counterparts at comparable scales, and reveal that reasoning-enhanced models (e.g., DeepSeek-R1, Qwen3-Thinking) paradoxically underperform on structured extraction tasks due to high parse failure rates.

## Repository Structure

```
.
├── v2_eval_pipeline.py          # Main evaluation pipeline
├── v2_generate_all_figures.py   # Figure and table generation
├── gen_sample_data.py           # Script to generate de-identified sample data
├── sample_data.json             # 8 de-identified example cases
├── requirements.txt             # Python dependencies
└── README.md
```

## Quick Start

### Prerequisites

- **llama.cpp** (b8826 or later) with `llama-server` binary
- GGUF model files in Q4_K_M quantization (download from HuggingFace)
- Python 3.10+ with dependencies from `requirements.txt`

### Installation

```bash
pip install -r requirements.txt
```

### Usage

```bash
# Smoke test -- 2 samples per available model, validates pipeline
python v2_eval_pipeline.py --smoke

# Full evaluation -- 10 runs, all available models
python v2_eval_pipeline.py 10 0

# Re-test only thinking/reasoning models
python v2_eval_pipeline.py 10 0 --rethink

# Generate all 6 figures and 3 tables from merged results
python v2_generate_all_figures.py
```

### Configuration

Edit the following variables at the top of `v2_eval_pipeline.py`:

| Variable | Description |
|----------|-------------|
| `LLAMACPP_SERVER_EXE` | Path to `llama-server` binary |
| `MODELS_DIR` | Directory containing `.gguf` model files |
| `SERVER_THREADS` | Number of CPU threads |
| `NUM_RUNS` | Number of independent evaluation runs (default: 10) |
| `MAX_TOKENS` | Maximum generation tokens (default: 2048) |

## Experimental Design

### Dual-Condition Framework

| Condition | Input Text | Task | Clinical Scenario |
|-----------|-----------|------|-------------------|
| **A (Extraction)** | Complete pathology report with diagnostic conclusions | Extract and standardize numerical grades | Practical: LLM as data extraction tool |
| **B (Reasoning)** | Microscopy description only, without diagnostic conclusions | Infer grades from morphological evidence | Cognitive: LLM pathological reasoning test |

### Models Evaluated

28 open-source LLMs from 8 model families, uniformly quantized to Q4_K_M:

| Family | Models | Parameter Range |
|--------|--------|----------------|
| Alibaba Qwen3 | 8 (incl. Dense, MoE, Instruct, Thinking) | 4B -- 32B |
| DeepSeek-R1 | 5 (distilled from R1 671B) | 7B -- 32B |
| Google Gemma / MedGemma | 5 | 4B -- 27B |
| Zhipu GLM-4 | 2 | 9B, 32B |
| OpenAI GPT-OSS | 1 x 3 reasoning modes | 20B (3.6B active) |
| Meta Llama | 2 | 3B, 8B |
| Tencent Hunyuan | 2 | 4B, 7B |
| Others | 3 (Granite, Moonlight) | 4B -- 16B |

All models evaluated with `temperature=0`, `max_tokens=2048`, across 10 independent runs with randomized few-shot examples.

### Evaluation Metrics

- Macro F1, Weighted F1, Accuracy (mean +/- SD across 10 runs)
- Macro Precision, Macro Recall, Macro Specificity
- Per-class Precision, Recall, Specificity, F1
- Parse error rate (categorized: empty_output, not_integer, not_in_labels, call_error)
- Per-sample predictions (patient_id, true_label, predicted_label)

## Data

Due to patient privacy and institutional IRB restrictions, the full clinical dataset (268 MASLD liver biopsy reports) cannot be publicly shared. It is available from the corresponding author upon reasonable request.

For **reproducibility verification**, `sample_data.json` contains 8 de-identified example cases, each with:

- `sample_id` — anonymous identifier
- `fibrosis_stage` (0-4), `inflammation_grade` (0-4), `steatosis_grade` (1-3) — ground truth labels
- `pathology_text_with_diagnosis` — complete pathology report (Condition A input)
- `pathology_text_without_diagnosis` — microscopy description only (Condition B input)

To verify pipeline functionality with the sample data:

```bash
python gen_sample_data.py                          # Generate sample_data.json
python v2_eval_pipeline.py --smoke                  # Quick test with available models
```

## Results Summary

Key findings from the full 28-model evaluation:

| Metric | Best Model | Score |
|--------|-----------|-------|
| Fibrosis Extraction (A) | Qwen3-14B | Macro F1 = 0.928 |
| Steatosis Extraction (A) | GLM-4-32B | Macro F1 = 0.961 |
| Fibrosis Reasoning (B) | Qwen3-14B | Macro F1 = 0.503 |
| Mean Inference Gap (A-B) | — | ΔMF1 ≈ 0.30 |
| Chinese vs English @ 8B | Qwen3-8B vs Llama-3.1-8B | ΔMF1 = +0.33 |

## Citation

If you use this code or data, please cite:

```bibtex
@article{xxx2025extraction,
  title={The extraction-reasoning gap: A dual-condition benchmark of 28 open-source LLMs for grading liver pathology reports},
  author={Xu Chengying et al.},
  journal={Scientific Reports},
  year={2025},
  doi={10.5281/zenodo.20162415}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
