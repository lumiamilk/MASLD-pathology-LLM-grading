> # Wilcoxon Signed-Rank Test: Condition A vs B (Macro F1)
>
> Method: Two-sided paired Wilcoxon signed-rank test, comparing Condition A and Condition B across n = 28 model configurations per task.
>
> Multiple comparison correction: Benjamini-Hochberg FDR correction across the three task-level comparisons.
>
> Zero-difference handling: zero_method='wilcox'.
>
> ## Results
>
> | Task         | n    | W    | Mean A | Mean B | Gap    | Raw P    | BH-adjusted P | Significant |
> | ------------ | ---- | ---- | ------ | ------ | ------ | -------- | ------------- | ----------- |
> | fibrosis     | 28   | 0    | 0.7251 | 0.3983 | 0.3268 | 7.45e-09 | 1.12e-08      | Yes         |
> | inflammation | 28   | 2    | 0.4580 | 0.2375 | 0.2205 | 2.24e-08 | 2.24e-08      | Yes         |
> | steatosis    | 28   | 0    | 0.8519 | 0.5978 | 0.2541 | 7.45e-09 | 1.12e-08      | Yes         |
>
> ## Interpretation
>
> All three tasks showed statistically significant differences after Benjamini-Hochberg correction. For fibrosis and steatosis, all 28 model configurations performed better under Condition A than Condition B. For inflammation, 26 of 28 model configurations favored Condition A.
