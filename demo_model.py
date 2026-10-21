"""Input conversion and inference for the local project demonstration."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CLASS_NAMES = {0: "Never used", 1: "Former user", 2: "Recent user"}
CLASS_MEANINGS = {
    0: "Estimated category: never used cannabis.",
    1: "Estimated category: last used cannabis more than one year ago.",
    2: "Estimated category: used cannabis within the last year.",
}
AGE = {"18–24": -0.95197, "25–34": -0.07854, "35–44": 0.49788,
       "45–54": 1.09449, "55–64": 1.82213, "65+": 2.59171}
EDUCATION = {"Left before 16": -2.43591, "Left at 16": -1.73790,
             "Left at 17": -1.43719, "Left at 18": -1.22751,
             "Some college, no degree": -0.61113,
             "Professional certificate/diploma": -0.05921,
             "University degree": 0.45468, "Masters degree": 1.16365,
             "Doctorate": 1.98437}
# Each tuple contains a label, a low example, and a high example.
TRAITS = {
    "nscore": ("Neuroticism", "Usually calm under pressure", "Often worries or feels stressed"),
    "escore": ("Extraversion", "Prefers quiet time or small groups", "Enjoys socialising and lively groups"),
    "oscore": ("Openness", "Prefers familiar ideas and routines", "Enjoys new ideas and creative experiences"),
    "ascore": ("Agreeableness", "More competitive or sceptical", "More cooperative and trusting"),
    "cscore": ("Conscientiousness", "More spontaneous and less structured", "More organised and dependable"),
    "impulsive": ("Impulsivity", "Usually thinks before acting", "Often acts on the spur of the moment"),
    "ss": ("Sensation seeking", "Prefers familiar, quiet activities", "Enjoys excitement and new sensations"),
}
LEVEL_LABELS = {1: "1 · Low", 2: "2 · Below middle", 3: "3 · Middle",
                4: "4 · Above middle", 5: "5 · High"}


def load_resources():
    """Load local fitted artifacts; never fit models on submitted inputs."""
    svm = joblib.load(ROOT / "data/processed/svm_pipeline.joblib")
    with np.load(ROOT / "data/processed/logistic_model.npz", allow_pickle=False) as saved:
        logistic = {key: saved[key].copy() for key in saved.files}
    training = pd.read_csv(ROOT / "data/processed/X_train.csv", index_col="id")
    names = svm.named_steps["preprocessing"].get_feature_names_out()
    if not np.array_equal(names, logistic["feature_names"]):
        raise ValueError("The models have different feature orders. Rerun the project notebook.")
    return svm, logistic, training


def level_values(training):
    """Five illustrative levels use training-only 10/30/50/70/90th percentiles.

    These are a demonstration convention, not a validated questionnaire conversion.
    Nearest quantiles keep the values on the dataset's observed score scale.
    """
    return {name: training[name].quantile([.1, .3, .5, .7, .9], interpolation="nearest").to_numpy()
            for name in TRAITS}


def make_input(demographics, scores, columns):
    values = dict(demographics)
    values["age"] = AGE[values["age"]]
    values["education"] = EDUCATION[values["education"]]
    values.update(scores)
    frame = pd.DataFrame([values], columns=columns)
    if frame.isna().any().any() or not np.isfinite(frame.select_dtypes("number")).all().all():
        raise ValueError("Please supply all 12 features with finite numeric scores.")
    return frame


def predict_both(frame, svm, logistic):
    transformed = svm.named_steps["preprocessing"].transform(frame)
    scores = transformed @ logistic["weights"].T + logistic["biases"]
    # Match the notebook's clipped sigmoid and one-vs-rest argmax exactly.
    scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -500, 500)))
    lr_predictions = logistic["classes"][np.argmax(scores, axis=1)]
    return lr_predictions, svm.predict(frame)
