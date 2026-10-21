# Cannabis consumption prediction

Machine Learning course project · **DA380A, Group 07**

Predict self-reported cannabis-use recency from 12 demographic and
personality/behavioural features. The project includes data preparation,
**logistic regression implemented from scratch**, a **library RBF SVM**, and a
**Streamlit app showing both models' predictions side by side**.

- [Project notebook](notebooks/ML_project.ipynb): executed code, explanations, exploration, training and evaluation.
- [Algorithm 2 results](reports/algorithm2_summary.md): SVM settings, results and discussion.
- [Model comparison](reports/model_comparison.csv): saved evaluation metrics.

## Quick start: run the app

From the project folder in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

Open **http://localhost:8501** in your browser. Stop the app with **Ctrl+C** in
the terminal. If `.venv` already exists, skip its creation. The local app and
interaction checks were verified with Python 3.12; the original preparation
was also run with Python 3.14. Dependency versions are in
[requirements.txt](requirements.txt), and actual training versions are recorded
in the preparation and SVM metadata files.

The app loads the included fitted models. You do not need to run the notebook
before opening it. If model artifacts are missing, run all notebook cells,
including the export cells, and restart the app.

## Using the prediction form

1. Choose **Guided levels (1–5)** or **Exact dataset scores**.
2. Select age band, gender, highest education, country of residence and ethnicity.
3. Enter the seven personality/behavioural scores.
4. Click **Compare both models**.
5. Read each model's predicted category and its meaning. Expand **See the exact
   scores used** to inspect the submitted model inputs.

The output shows **Algorithm 1: logistic regression** and **Algorithm 2: RBF SVM**
in separate cards and indicates whether they agree. Both receive the same
profile. No confidence percentages are shown: the scratch model's independent
probabilities do not sum to one, and the SVM has no calibrated probability model.
Inputs are processed locally and are not saved by the app.

### What do levels 1–5 mean?

**1 = low, 3 = middle, 5 = high** for the named trait. A higher value is not
necessarily better or worse. Each slider includes examples:

| Trait | Example at 1 (low) | Example at 5 (high) |
|---|---|---|
| Neuroticism | Usually calm under pressure | Often worries or feels stressed |
| Extraversion | Prefers quiet time or small groups | Enjoys socialising and lively groups |
| Openness | Prefers familiar ideas and routines | Enjoys new ideas and creative experiences |
| Agreeableness | More competitive or sceptical | More cooperative and trusting |
| Conscientiousness | More spontaneous and less structured | More organised and dependable |
| Impulsivity | Usually thinks before acting | Often acts on the spur of the moment |
| Sensation seeking | Prefers familiar, quiet activities | Enjoys excitement and new sensations |

These are **illustrative demo levels, not a validated personality questionnaire**.
They map to the nearest observed training scores at the 10th, 30th, 50th, 70th
and 90th percentiles. Level 5 is the highest demo level, not the maximum possible
dataset score. Predictions from these inputs include this additional approximation.

Use **Exact dataset scores** if you have the seven UCI quantified scores.
Enter values **before our model's standardisation**. Negative values are valid.
Do not enter raw questionnaire totals or already-standardised model inputs.
The form shows training ranges and flags values outside those ranges.

## Prediction target

| Class | Meaning | Original cannabis codes |
|---|---|---|
| 0 · Never used | No reported cannabis use | CL0 |
| 1 · Former user | Last used more than one year ago | CL1, CL2 |
| 2 · Recent user | Used within the last year | CL3, CL4, CL5, CL6 |

This is a three-class classification problem about **use recency**, not
consumption amount, dependence or future use. A prediction does not establish
someone's actual history.

## Dataset and preparation

- Source: UCI Drug Consumption (Quantified), **1,885 respondents, 32 columns**.
- No missing values or exact duplicate responses were found.
- Eight respondents reporting use of fictitious Semeron were excluded by a
  documented response-quality rule, leaving **1,877 respondents**. This exclusion
  does not prove that all their other responses were false.
- Inputs: age, gender, education, country, ethnicity, neuroticism, extraversion,
  openness, agreeableness, conscientiousness, impulsivity and sensation seeking.
- ID and all drug-use responses, including the original cannabis column, are
  excluded from the predictors.
- Fixed **80/20 stratified split**, seed **42**: **1,501 training / 376 test** rows.
- Gender, country and ethnicity are one-hot encoded. Age, education and the seven
  scores are standardised, producing **25 numeric features**.
- Preprocessing is fitted on training data only and separately within each
  cross-validation fold. Legitimate extreme values are retained; no imputation,
  oversampling or feature selection is performed.

Training classes: **330 never (22.0%), 378 former (25.2%), 793 recent (52.8%)**.
Macro-F1 is the primary selection metric because it gives each class equal weight.

## Models and results

### Algorithm 1: logistic regression from scratch

Three one-versus-rest classifiers implemented with NumPy, Pandas and Matplotlib.
Each learns using batch gradient descent on binary cross-entropy with an L2
penalty. The class with the largest sigmoid output is selected.

Five-fold stratified validation selected **learning rate 0.5, L2 0.001,
1,000 iterations**. Validation macro-F1: **0.603 ± 0.023**.
The app loads the fitted weights from `data/processed/logistic_model.npz`;
it does not substitute a library logistic-regression model.

### Algorithm 2: RBF support vector machine

Scikit-learn `SVC` in a preprocessing pipeline. A 32-configuration search tunes
`C`, `gamma` and class weighting using the **exact same five folds** as Algorithm 1.
Preprocessing is refitted inside every fold. SVM settings are selected using
training data only; Algorithm 1's holdout results were already visible.

Selected: **C=100, gamma=0.001, class_weight=None**.
Validation macro-F1: **0.614 ± 0.018**. Full-training macro-F1: **0.641**.
The selected C and numeric gamma are at grid boundaries, so this is the best
tested configuration rather than a proven global optimum. Fold standard
deviations describe variability, not confidence intervals.

### Same-holdout comparison

| Metric | Majority baseline | Logistic regression | RBF SVM |
|---|---:|---:|---:|
| Accuracy | 0.527 | **0.694** | 0.691 |
| Macro-F1 | 0.230 | 0.614 | **0.624** |
| Former-user recall | 0.000 | 0.347 | **0.421** |

SVM finds **40 rather than 33 of 95 former users**, while making one fewer correct
prediction overall (**260 versus 261 of 376**). It improves the primary metric
modestly, with reduced recall for the other classes. These results do not establish
universal superiority. The notebook includes per-class metrics and side-by-side
confusion matrices.

## Rerun the notebook

```powershell
.\.venv\Scripts\python.exe -m jupyterlab
```

Open `notebooks/ML_project.ipynb`, choose the `.venv` kernel, and select **Run All**.
VS Code can also run the notebook with that environment. The raw file is included
and checksum verified; downloading happens only if it is absent. Rerunning
regenerates processed data, models, reports and figures with the fixed split.
Restart the app after regenerating models so it reloads its cached artifacts.

For model development, use `X_train.csv` with preprocessing fitted inside each
fold. The `*_ready.csv` files are already encoded and scaled; do not transform
them again or use them for tuning across folds. Load `id` as the index, never
as a predictor, and preserve X/y row alignment. Reserve the same holdout for
final comparison after fixing model settings.

## Run the checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The checks cover both models matching all 376 saved holdout predictions,
demographic conversion, training-derived score levels, and submission of both
guided and exact-score forms. Browser-based visual inspection was unavailable
in the development session; form interactions were checked with Streamlit AppTest.

## Repository contents

| Path | Purpose |
|---|---|
| `app.py` | Streamlit input form and side-by-side results |
| `demo_model.py` | Input conversion, trait descriptions and inference |
| `.streamlit/config.toml` | App theme and usage-statistics setting |
| `notebooks/ML_project.ipynb` | Data preparation, both algorithms, evaluation and exports |
| `data/raw/drug_consumption.data` | Unchanged source data |
| `data/processed/X_train.csv`, `X_test.csv` | Twelve predictors before fitted preprocessing |
| `data/processed/*_ready.csv`, `y_*.csv` | Transformed inputs and aligned target labels |
| `data/processed/preprocessor.joblib` | Training-fitted scaler and encoder |
| `data/processed/logistic_model.npz` | Fitted scratch-model weights, biases, classes and feature order |
| `data/processed/svm_pipeline.joblib` | Fitted SVM with its preprocessing |
| `data/processed/cleaned_data.csv`, `split_membership.csv` | Retained responses and split audit |
| `reports/data_dictionary.csv`, `excluded_respondents.csv` | Column meanings and cleaning audit |
| `reports/preparation_metadata.json`, `svm_metadata.json` | Preparation/training settings and runtime versions |
| `reports/training_*.csv`, `figures/` | Training exploration and saved plots |
| `reports/svm_cv_results.csv` | Full SVM tuning results |
| `reports/svm_test_classification_report.csv` | Per-class SVM evaluation |
| `reports/model_comparison.csv`, `model_test_predictions.csv` | Model metrics and holdout predictions |
| `reports/phase1_summary.md`, `algorithm2_summary.md` | Preparation and SVM summaries |
| `tests/test_demo.py` | Inference parity and app interaction checks |
| `requirements.txt` | Python dependencies |

## Interpretation and source

The survey is self-reported and unevenly sampled, with small demographic groups.
Results describe this sample, not population prevalence or causal effects.
Country and ethnicity may reflect recruitment patterns and confounding factors.
The released attributes were quantified upstream; that transformation cannot be
refitted from the released file. `meth` means methadone.

Fehrman, E., Egan, V., & Mirkes, E. (2015). *Drug Consumption (Quantified)*.
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/373/drug-consumption-quantified).
[DOI: 10.24432/C5TC7S](https://doi.org/10.24432/C5TC7S). Dataset licence: **CC BY 4.0**.
