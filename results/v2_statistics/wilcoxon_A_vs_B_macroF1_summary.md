# Wilcoxon Signed-Rank Test: Condition A vs B (Macro F1)

**Method**: Two-sided paired Wilcoxon signed-rank test, comparing Condition A (extraction) and Condition B (reasoning) across n=28 independently evaluated model configurations per task.
**Multiple comparison correction**: Benjamini-Hochberg FDR (alpha = 0.05) across the three tasks.
**Zero-difference handling**: zero_method='wilcox'.

## Results

| Task | n | W | Mean A | Mean B | Gap | Raw P | BH-adjusted P | Significant |
|------|---|---|--------|--------|-----|-------|---------------|-------------|
| fibrosis | 28 | 0 | 0.7251 | 0.3983 | 0.3268 | 7.45e-09 | 1.12e-08 | Yes |
| inflammation | 28 | 2 | 0.4580 | 0.2375 | 0.2205 | 2.24e-08 | 2.24e-08 | Yes |
| steatosis | 28 | 0 | 0.8519 | 0.5978 | 0.2541 | 7.45e-09 | 1.12e-08 | Yes |

## Interpretation

All three tasks showed statistically significant differences after Benjamini-Hochberg correction (adjusted P < 0.05). For fibrosis and steatosis, **all 28 model configurations** performed better under Condition A than Condition B (W = 0, P < 1e-8). For inflammation, 26 out of 28 models favoured Condition A (W = 2, P < 2.5e-8).

## Paper Sentence — Results

> At the model-configuration level (n = 28), two-sided Wilcoxon signed-rank tests confirmed higher macro F1 under Condition A than Condition B after Benjamini-Hochberg correction across the three tasks (fibrosis: W = 0, adjusted P = 1.12e-08; inflammation: W = 2, adjusted P = 2.24e-08; steatosis: W = 0, adjusted P = 1.12e-08). Complete rankings are provided in Supplementary Tables S1-S3.

## Paper Sentence — Methods

> Because the same model configuration was evaluated under both conditions, paired comparisons were performed at the model-configuration level using two-sided Wilcoxon signed-rank tests with an alpha level of 0.05. Where multiple task-level comparisons were considered jointly, false-discovery-rate control was applied using the Benjamini-Hochberg procedure.
