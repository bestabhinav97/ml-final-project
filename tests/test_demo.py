"""Check inference parity with the notebook and the two input flows."""
import unittest

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from demo_model import ROOT, TRAITS, load_resources, level_values, make_input, predict_both


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svm, cls.logistic, cls.training = load_resources()

    def test_both_models_match_all_saved_holdout_predictions(self):
        inputs = pd.read_csv(ROOT / "data/processed/X_test.csv", index_col="id")
        expected = pd.read_csv(ROOT / "reports/model_test_predictions.csv", index_col="id")
        self.assertTrue(inputs.index.equals(expected.index))
        lr, svm = predict_both(inputs, self.svm, self.logistic)
        np.testing.assert_array_equal(lr, expected.logistic_prediction)
        np.testing.assert_array_equal(svm, expected.svm_prediction)

    def test_demographic_conversion_matches_training_row(self):
        row = self.training.iloc[0]
        from demo_model import AGE, EDUCATION
        background = {k: row[k] for k in ["gender", "country", "ethnicity"]}
        background["age"] = next(k for k, v in AGE.items() if v == row.age)
        background["education"] = next(k for k, v in EDUCATION.items() if v == row.education)
        result = make_input(background, {k: row[k] for k in TRAITS}, self.training.columns)
        pd.testing.assert_frame_equal(result.reset_index(drop=True), self.training.iloc[[0]].reset_index(drop=True))

    def test_level_mapping_uses_monotonic_observed_training_scores(self):
        for name, values in level_values(self.training).items():
            self.assertTrue(np.all(np.diff(values) >= 0))
            self.assertTrue(np.isin(values, self.training[name]).all())

    def test_guided_and_exact_forms_show_both_results(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.select_slider), 7)
        app.select_slider(key="level_ss").set_value(5)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 2)
        for metric in app.metric:
            self.assertIn(metric.value, ["Never used", "Former user", "Recent user"])
        app.radio[0].set_value("Exact dataset scores").run()
        self.assertEqual(len(app.number_input), 7)
        # An extrapolated score still predicts, with a visible range message.
        app.number_input(key="exact_ss").set_value(10.0)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 2)
        self.assertTrue(any("Outside the training range" in w.value for w in app.warning))


if __name__ == "__main__":
    unittest.main()
