# Cannabis Use Classification

**Machine Learning course project | DA380A | Group 07**

In this project, we used machine learning to predict how recently someone had used cannabis. We worked with demographic and personality data and compared two models: Logistic Regression, which we implemented from scratch, and an RBF Support Vector Machine (SVM) using scikit-learn.

The code, training process and evaluation are in [`notebooks/ML_project.ipynb`](notebooks/ML_project.ipynb).

## Dataset

We used the **Drug Consumption (Quantified)** dataset from the UCI Machine Learning Repository. It originally had 1,885 responses. After excluding 8 responses based on the fictitious drug *Semeron*, we had 1,877 responses left.

We used 12 features, including age, education and personality scores, to predict three classes:

| Class | Meaning |
|---|---|
| 0 | Never used cannabis |
| 1 | Former user (last use over a year ago) |
| 2 | Recent user (used within the last year) |

We split the data into **80% training (1,501 people)** and **20% testing (376 people)**.

## Models

- **Logistic Regression:** Implemented from scratch using a one-vs-rest approach for the three classes.
- **SVM:** Implemented with scikit-learn using an RBF kernel.

Both models were compared using accuracy, precision, recall, F1-score and confusion matrices. We also used five-fold cross-validation during model selection.

## Results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Logistic Regression | 69.4% | 0.614 |
| SVM | 69.1% | 0.624 |

Both models gave similar results. Logistic Regression got one more prediction correct on the test set, while SVM had a higher Macro F1-score. SVM also predicted more former users correctly, which was the class both models struggled with the most.

The full results, including the confusion matrices and error analysis, are in the project notebook.

## How to run the notebook

1. Download or clone the repository.
2. Install the packages listed in `requirements.txt`:

   ```bash
   python -m pip install -r requirements.txt
   ```

   On some Macs, use `python3` instead of `python`.
3. Open [`notebooks/ML_project.ipynb`](notebooks/ML_project.ipynb) in VS Code or Jupyter, select your Python environment, and run the cells in order.

**Note:** Some cells use paths such as `../reports/`. Make sure the notebook's working directory is `notebooks/` when running them.

## Project files

| Path | What's there |
|---|---|
| `notebooks/ML_project.ipynb` | Data preparation, training, evaluation and comparison |
| `data/raw/` | Original dataset |
| `data/processed/` | Cleaned data, train/test split and saved model files |
| `reports/` | Saved metrics, predictions and figures |
| `requirements.txt` | Python packages needed for the project |

## Dataset source

UCI Machine Learning Repository — *Drug Consumption (Quantified)*.s, E. (2015). *Drug Consumption (Quantified)*.
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/373/drug-consumption-quantified).
[DOI: 10.24432/C5TC7S](https://doi.org/10.24432/C5TC7S). Dataset licence: **CC BY 4.0**.
