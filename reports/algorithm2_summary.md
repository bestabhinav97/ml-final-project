## Algorithm 2: RBF support vector machine (library)

Implemented in Part 3 of the notebook using a scikit-learn pipeline. The 32-setting
search tunes C, gamma, and class weighting using the exact five validation folds
from Algorithm 1. Scaling and encoding are refitted inside each fold. Macro-F1 is
the selection metric; no holdout results are used for tuning.

Selected: `C=100`, `gamma=0.001`, `class_weight=None`. Validation macro-F1 is
**0.614 +/- 0.018**, full-training macro-F1 **0.641**.

| Holdout metric | Logistic regression | RBF SVM |
|---|---:|---:|
| Accuracy | 0.694 | 0.691 |
| Macro-F1 | 0.614 | 0.624 |
| Former-user recall | 0.347 | 0.421 |

SVM improves the main metric and finds 40 rather than 33 former users, but makes
one fewer correct prediction overall. This is a modest trade-off, not conclusive
evidence of superiority. The selected C/gamma are at grid boundaries, so these
are the best tested settings rather than a proven global optimum.

Outputs: `reports/svm_cv_results.csv`, `reports/svm_metadata.json`,
`reports/svm_test_classification_report.csv`, `reports/model_comparison.csv`,
`reports/model_test_predictions.csv`, and comparison figures in `reports/figures/`.
The fitted preprocessing/model pipeline is `data/processed/svm_pipeline.joblib`.
Actual Algorithm 2 runtime versions are recorded in its metadata; the original
preparation metadata describes the earlier preparation environment.
