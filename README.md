# Cannabis consumption prediction (DA380A, Group 07)

**Data preparation and Algorithm 1 are complete.** Start with the executed, commented
[project notebook](notebooks/ML_project.ipynb). It explains loading, cleaning,
exploration, target creation, encoding, scaling, the train/test split, and the
**one-vs-rest logistic regression implemented from scratch** (NumPy/Pandas/Matplotlib
only), including 5-fold cross-validation and the final test evaluation.
Algorithm 2 (library model) is still to be added.

## What are we predicting?

Predict self-reported cannabis-use recency from 12 demographic and
personality/behavioral attributes. This is a **three-class classification** task:

| Answer | Meaning | Original UCI cannabis codes |
|---|---|---|
| 0 | Never used | CL0 |
| 1 | Former user: last used more than one year ago | CL1, CL2 |
| 2 | Recent user: used within the last year | CL3, CL4, CL5, CL6 |

This target describes recency, not dependence, amount consumed, or future use.

## Files teammates should use

All modeling files are in [`data/processed/`](data/processed/).
**X means inputs; y means the correct answers.**

| Purpose | Input file | Answer file | Rows |
|---|---|---|---|
| Train the models | `X_train_ready.csv` | `y_train.csv` | 1,501 |
| Final evaluation | `X_test_ready.csv` | `y_test.csv` | 376 |

The `ready` files already contain **25 numeric feature columns**, after encoding
and scaling. **Do not scale or encode them again.** The `id` column is only a row
identifier: load it as the index, never as a predictor. Preserve the alignment
between each X file and its corresponding y file.

From the repository root:

```python
import pandas as pd

X_train = pd.read_csv("data/processed/X_train_ready.csv", index_col="id")
X_test = pd.read_csv("data/processed/X_test_ready.csv", index_col="id")
y_train = pd.read_csv("data/processed/y_train.csv", index_col="id")["cannabis_class"]
y_test = pd.read_csv("data/processed/y_test.csv", index_col="id")["cannabis_class"]

assert X_train.index.equals(y_train.index)
assert X_test.index.equals(y_test.index)
assert X_train.columns.equals(X_test.columns)

# Use training inputs/answers for fitting. Reserve testing for final evaluation.
# If your from-scratch implementation needs NumPy arrays, use .to_numpy().
```

## What preparation was done?

1. Loaded the original UCI file: **1,885 respondents and 32 columns**.
2. Checked missing values, duplicate responses, IDs, finite numbers, and valid
   demographic/consumption codes. No missing values or exact duplicate responses
   were found.
3. Excluded **eight respondents** who reported using fictitious Semeron, leaving
   **1,877 respondents**. This is an explicit response-quality rule; it does not
   prove all their other answers were false. The excluded IDs are recorded in
   [`reports/excluded_respondents.csv`](reports/excluded_respondents.csv).
4. Created the three cannabis classes shown above. Excluded ID and all drug-use
   response columns from predictors, including the original cannabis response.
5. Made an **80/20 stratified split with random seed 42**. Stratification keeps
   class proportions approximately similar. Detailed exploration uses training only.
6. Fitted encoding/scaling **on training data only**, then applied the same fitted
   transformation to the test data.
7. Exported the datasets, fitted transformer, audit records, tables and charts.
   Checks passed for split overlap, label alignment, reproducibility, training-only
   scaling, and exported-file round trips.

No missing-value imputation, oversampling, class weighting, feature selection, or
outlier deletion was performed. Legitimate extreme scores were retained.

### How are the inputs encoded?

- **Gender, country, ethnicity:** one-hot encoding. Each category has a column
  containing 1 if it applies and 0 otherwise; e.g. `country_UK`. This avoids treating
  arbitrary country codes as meaningful numeric distances.
- **Age, education, and seven personality/behavioral scores:** UCI's quantified
  numeric values, standardized using training means and standard deviations.
  Age represents an age band, not literal years.
- **Answers:** integers 0, 1, 2, as defined above.

Training class counts are **330 never, 378 former, 793 recent**. Recent users make up
about 53% of training data, so accuracy alone is insufficient for comparison.

## Algorithm 1: logistic regression from scratch (one vs rest)

Implemented using only NumPy, Pandas and Matplotlib.

- **Model:** three yes/no logistic regressions (never / former / recent vs the rest);
  the predicted class is the one with the highest probability.
- **Learning:** batch gradient descent on cross entropy loss with an L2 penalty.
- **Validation:** stratified 5 fold cross validation inside the training set.
  The preprocessor is cloned and re fitted inside every fold so no
  validation data leaks into the scaling. The grid searched `lr` and `l2`;
  the chosen settings were `lr = 0.5`, `l2 = 0.001`, 1000 iterations.
- **Metrics:** confusion matrix, accuracy, per-class precision/recall/F1 and
  macro F1 (main metric) compared with a majority class baseline.
- **Test result (used once):** accuracy about 0.69 and macro F1 about 0.61
  (baseline: 0.53 and 0.23). "Former user" is the hardest class (recall about 0.35).

## What the modeling teammate does next

1. Implement the required logistic regression **from scratch**, supporting three
   classes (e.g. softmax or one-vs-rest). The input matrices contain no intercept
   column; add one if your implementation requires it. All one-hot indicator
   columns are retained; account for redundancy, for example with regularization.
2. Implement a second suitable classifier. The course brief allows a library
   implementation for the second method.
3. Tune and choose models using **training data only**. For cross-validation, use
   `X_train.csv` (the 12 predictors before fitted transformations), clone the
   preprocessor, and fit it separately inside each fold. Using the `ready` matrix
   across validation folds would let validation data influence preprocessing.
4. After finalizing choices, fit preprocessing and models on all training data,
   then evaluate on the reserved test set. Both models must use the same holdout.
5. Compare accuracy, **macro-F1**, per-class precision/recall, and a confusion matrix.
   Include the comparison and limitations in the report and presentation.

For fold-specific preprocessing, the saved transformer can be cloned without
carrying over its fitted state:

```python
import joblib
from sklearn.base import clone

X_train_before_transform = pd.read_csv("data/processed/X_train.csv", index_col="id")
template = joblib.load("data/processed/preprocessor.joblib")

# Inside each CV fold, use positional fold indices from the splitter:
# fold_preprocessor = clone(template)
# X_fit = fold_preprocessor.fit_transform(X_train_before_transform.iloc[fit_indices])
# X_valid = fold_preprocessor.transform(X_train_before_transform.iloc[valid_indices])
```

Use the recorded dependency versions for loading the saved transformer. Do not
mix `X_train.csv` (12 columns with strings) with `X_test_ready.csv` (25 numeric
columns); apply the same preprocessing representation to both splits.

## Setup and rerun

Verified with **Python 3.14**. From PowerShell:

```powershell
git clone https://github.com/bestabhinav97/ml-final-project.git
cd ml-final-project
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyterlab
```

Open `notebooks/ML_project.ipynb`, select the `.venv` kernel, and choose
**Run All**. VS Code can also open the notebook with that kernel. The notebook is
self-contained; no notebook-generation script is needed. The raw data is included
and checksum verified; downloading happens only if the file is absent. Rerunning
overwrites generated outputs with the deterministic split described above.

## Repository contents

| Path | Purpose |
|---|---|
| `notebooks/ML_project.ipynb` | Executed notebook: data preparation (Phase 1) and logistic regression from scratch (Algorithm 1) |
| `data/raw/drug_consumption.data` | Unchanged source file |
| `data/processed/*_ready.csv` | Model-ready training/test inputs |
| `data/processed/y_*.csv` | Target labels |
| `data/processed/X_train.csv`, `X_test.csv` | Inputs before fitted preprocessing; for CV/reuse |
| `data/processed/preprocessor.joblib` | Fitted training-only scaler and encoder |
| `data/processed/cleaned_data.csv` | Retained responses and target, for reference; do not use all columns as X |
| `data/processed/split_membership.csv` | IDs and fixed split membership |
| `reports/data_dictionary.csv` | Column roles and meanings |
| `reports/excluded_respondents.csv` | Audit of the cleaning exclusion |
| `reports/preparation_metadata.json` | Source checksum, split settings, features and versions |
| `reports/training_*.csv`, `reports/figures/` | Training-only exploration tables and charts |
| `reports/phase1_summary.md` | Short preparation results summary |
| `requirements.txt` | Python dependencies |

## Interpretation and source

The survey is self-reported and unevenly sampled, with small demographic groups.
Results describe this sample, not population prevalence or causal effects. The
source quantified attributes before release; that upstream transformation cannot
be refitted using only the released file. `meth` means methadone.

Fehrman, E., Egan, V., & Mirkes, E. (2015). *Drug Consumption (Quantified)*.
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified).
[DOI: 10.24432/C5TC7S](https://doi.org/10.24432/C5TC7S). Dataset license: **CC BY 4.0**.

