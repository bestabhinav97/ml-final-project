"""Run with: py -m streamlit run app.py"""
import pandas as pd
import streamlit as st

from demo_model import (AGE, EDUCATION, TRAITS, LEVEL_LABELS, CLASS_NAMES,
                        CLASS_MEANINGS, load_resources, level_values, make_input,
                        predict_both)

st.set_page_config(page_title="Cannabis recency · Model lab", page_icon="◈", layout="wide")


@st.cache_resource
def resources():
    return load_resources()


st.caption("DA380A · GROUP 07 · INTERACTIVE PROJECT DEMO")
st.title("One profile. Two predictions.")
st.write("Explore how logistic regression and an RBF support vector machine estimate cannabis-use recency from the same 12 features.")

try:
    svm, logistic, training = resources()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Could not load the model artifacts: {exc}")
    st.info("Run all cells in notebooks/ML_project.ipynb, then restart this app.")
    st.stop()

with st.sidebar:
    st.header("About this demo")
    st.write("**Never used** · no reported use\n\n**Former user** · last use over a year ago\n\n**Recent user** · use within the last year")
    st.divider()
    st.write("Both models receive the same profile. Their answers may differ because they learn different decision boundaries.")
    st.caption("Educational estimates from a self-reported survey. They do not establish a person's actual history, dependence, or future use.")
    st.caption("Inputs are processed locally and are not saved by this app.")
    st.link_button("Dataset and score documentation ↗", "https://archive.ics.uci.edu/dataset/373/drug-consumption-quantified")

mode = st.radio("How would you like to enter scores?", ["Guided levels (1–5)", "Exact dataset scores"], horizontal=True)
guided = mode == "Guided levels (1–5)"
if guided:
    st.info("Demo scale: 1 = low, 3 = middle, 5 = high. These levels map to representative training-data scores; they are not a personality test. Choose exact scores if you already have UCI-format values.")
else:
    st.info("Enter UCI quantified scores, before our model's standardisation. Negative values are valid; zero is near the middle of the released score scale. Do not enter raw questionnaire totals or already-standardised model inputs.")

levels = level_values(training)
with st.form("profile"):
    st.subheader("01 / Your background")
    left, right = st.columns(2)
    with left:
        age = st.selectbox("Age band", list(AGE), index=1, help="Choose an age band, not a numeric dataset code.")
        gender = st.selectbox("Gender", list(training.gender.unique()), help="The source survey records only these two categories.")
        education = st.selectbox("Highest education", list(EDUCATION), index=6)
    with right:
        country = st.selectbox("Country of residence", sorted(training.country.unique()), index=sorted(training.country.unique()).index("UK"))
        ethnicity = st.selectbox("Ethnicity", sorted(training.ethnicity.unique()), index=sorted(training.ethnicity.unique()).index("White"), help="Categories follow the source dataset.")
        st.caption("Country and ethnicity reflect an uneven survey sample. They are included to reproduce the project models.")

    st.subheader("02 / Personality & behaviour")
    st.caption("Example starting profile: age 25–34, university degree, with all seven traits at the middle level. Change the values to explore.")
    columns = st.columns(2)
    scores = {}
    chosen_levels = {}
    for i, (name, (label, low, high)) in enumerate(TRAITS.items()):
        with columns[i % 2]:
            if guided:
                value = st.select_slider(label, options=[1, 2, 3, 4, 5], value=3,
                                         format_func=lambda v: LEVEL_LABELS[v], key=f"level_{name}",
                                         help=f"1 (low): {low}. 5 (high): {high}.")
                chosen_levels[name] = value
                scores[name] = float(levels[name][value - 1])
                st.caption(f"1: {low}. 5: {high}.")
            else:
                scores[name] = st.number_input(label, value=float(training[name].median()),
                                               step=.01, format="%.5f", key=f"exact_{name}",
                                               help=f"Lower: {low}. Higher: {high}.")
                st.caption(f"{low} → {high}. Training range: {training[name].min():.3f} to {training[name].max():.3f}.")
    submitted = st.form_submit_button("Compare both models", type="primary", width="stretch")

if submitted:
    demographics = {"age": age, "gender": gender, "education": education,
                    "country": country, "ethnicity": ethnicity}
    try:
        frame = make_input(demographics, scores, training.columns)
        lr_prediction, svm_prediction = predict_both(frame, svm, logistic)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    st.subheader("03 / Model comparison")
    st.caption("Results for the profile submitted above. Submit again after changing inputs.")
    for column, model, prediction, detail in zip(
        st.columns(2), ["Logistic regression", "RBF support vector machine"],
        [int(lr_prediction[0]), int(svm_prediction[0])],
        ["Algorithm 1 · implemented from scratch", "Algorithm 2 · scikit-learn"]):
        with column:
            with st.container(border=True):
                st.caption(detail.upper())
                st.subheader(model)
                st.metric("Predicted category", CLASS_NAMES[prediction])
                st.write(CLASS_MEANINGS[prediction])
    if lr_prediction[0] == svm_prediction[0]:
        st.success("Both models predict the same category for this profile.")
    else:
        st.info("The models disagree on this profile. Agreement or disagreement does not tell us the person's actual history.")
    if guided:
        st.caption("These results use approximate demo levels, not measured questionnaire scores. Neither result is a verified fact about a person.")
    outside = [TRAITS[k][0] for k, v in scores.items() if v < training[k].min() or v > training[k].max()]
    if outside:
        st.warning("Outside the training range: " + ", ".join(outside) + ". The models are extrapolating for these values.")
    with st.expander("See the exact scores used"):
        st.dataframe(frame, hide_index=True, width="stretch")
        if guided:
            st.caption("Levels 1–5 map to the 10th, 30th, 50th, 70th and 90th percentiles of each training score, using the nearest observed value. Level 5 is the highest demo level, not the maximum possible score.")

with st.expander("Understand the scores and model results"):
    st.write("A high score means more of the named trait, not better or worse. For example, 5 in conscientiousness represents being more organised; 5 in impulsivity represents acting more spontaneously.")
    st.write("The original survey used questionnaire-derived, quantified scores. Our five-level inputs are an illustrative shortcut, not a validated replacement for those questionnaires.")
    st.write("We show category predictions without confidence percentages: the scratch model's independent probabilities do not sum to one, and the SVM has no calibrated probability model.")
    st.write("On the reserved test set, logistic regression achieved 69.4% accuracy / 0.614 macro-F1; SVM achieved 69.1% / 0.624. These are dataset-level metrics, not confidence in a particular prediction.")
    if guided:
        st.dataframe(pd.DataFrame({TRAITS[k][0]: levels[k] for k in TRAITS},
                                 index=[LEVEL_LABELS[i] for i in range(1, 6)]), width="stretch")
