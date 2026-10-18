# Phase 1 results

Source: https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified

- Raw respondents: 1,885; columns: 32.
- Missing values: 0; exact duplicate responses: 0.
- Semeron exclusions: 8; cleaned respondents: 1,877.
- Training/test rows: 1,501/376; stratified 80/20, seed 42.
- Target: 0=never (CL0), 1=former (CL1-CL2), 2=recent (CL3-CL6).
- Training class counts: {0: 330, 1: 378, 2: 793}.
- Predictors: 12 original, 25 after transformation.
- Nominal demographics one-hot encoded; nine numeric attributes standardized on training only.
- No imputation, outlier deletion, resampling or predictive model fitting.
- All preparation, leakage, reproducibility and file round-trip checks passed.

Limitations: self-report, uneven survey sampling, small demographic groups and source
quantification already applied before release. These are sample descriptions, not causal effects.
See the notebook for training-only tables, charts and explanations.
