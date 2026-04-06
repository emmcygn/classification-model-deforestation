"""Tests verifying temporal integrity of the deforestation risk model.

These tests guard against target leakage — the most dangerous failure mode
for a prospective risk model. If any feature encodes information from the
prediction period (2020-2022), the model is learning "what already happened"
instead of "what will happen."
"""

import pytest
import numpy as np
import pandas as pd
from core.ml.dataset import (
    load_dataset, prepare_features, _apply_temporal_split, validate_features,
    FEATURE_CUTOFF, TARGET_START, FEATURE_COLUMNS_FULL, FEATURE_MANIFEST,
)


@pytest.fixture
def palawan_df():
    return load_dataset(region="palawan")


@pytest.fixture
def all_regions_df():
    return load_dataset()


class TestTemporalSplit:
    """Verify _apply_temporal_split correctly separates features from target."""

    def test_high_risk_only_from_target_period(self, palawan_df):
        """high_risk=1 cells must have loss_year >= 20 (2020+)."""
        hr = palawan_df[palawan_df["high_risk"] == 1]
        assert (hr["loss_year"] >= TARGET_START).all(), (
            f"Found high_risk=1 cells with loss_year < {TARGET_START}: "
            f"{hr[hr['loss_year'] < TARGET_START]['loss_year'].value_counts().to_dict()}"
        )

    def test_pre_period_loss_only_from_feature_period(self, palawan_df):
        """pre_period_loss=1 cells must have loss_year in [1, 19]."""
        pre = palawan_df[palawan_df["pre_period_loss"] == 1]
        assert ((pre["loss_year"] > 0) & (pre["loss_year"] <= FEATURE_CUTOFF)).all(), (
            "pre_period_loss=1 cells found outside 2001-2019"
        )

    def test_no_overlap_between_target_and_feature_loss(self, palawan_df):
        """A cell cannot be both pre_period_loss=1 AND high_risk=1."""
        overlap = palawan_df[
            (palawan_df["pre_period_loss"] == 1) & (palawan_df["high_risk"] == 1)
        ]
        assert len(overlap) == 0, (
            f"{len(overlap)} cells have both pre_period_loss=1 and high_risk=1"
        )

    def test_temporal_split_recomputes_from_loss_year(self):
        """_apply_temporal_split must override CSV values from loss_year."""
        # Create a fake DF with deliberately wrong high_risk
        df = pd.DataFrame({
            "loss_year": [0, 5, 19, 20, 22],
            "high_risk": [1, 1, 1, 0, 0],  # all wrong
        })
        result = _apply_temporal_split(df)
        expected_hr = [0, 0, 0, 1, 1]
        assert list(result["high_risk"]) == expected_hr

        expected_pre = [0, 1, 1, 0, 0]
        assert list(result["pre_period_loss"]) == expected_pre


class TestFeatureManifest:
    """Verify the feature manifest guards against post-target features."""

    def test_all_training_features_are_allowed(self):
        """FEATURE_COLUMNS_FULL must only contain allowed features."""
        for col in FEATURE_COLUMNS_FULL:
            assert col in FEATURE_MANIFEST, f"Feature '{col}' not in manifest"
            _, _, allowed = FEATURE_MANIFEST[col]
            assert allowed, f"Feature '{col}' is in FEATURE_COLUMNS_FULL but marked not allowed"

    def test_banned_features_are_rejected(self):
        """validate_features must reject post-target features."""
        with pytest.raises(ValueError, match="not allowed"):
            validate_features(["exg_change_2018_2022"])

        with pytest.raises(ValueError, match="not allowed"):
            validate_features(["loss_year"])

        with pytest.raises(ValueError, match="not allowed"):
            validate_features(["high_risk"])

    def test_allowed_features_pass_validation(self):
        """validate_features must accept all FEATURE_COLUMNS_FULL."""
        validate_features(FEATURE_COLUMNS_FULL)  # should not raise


class TestFeatureLeakage:
    """Verify no feature column leaks target-period information."""

    def test_no_feature_strongly_correlates_with_target(self, palawan_df):
        """No feature should correlate with high_risk above |r|=0.55.

        This threshold is a pragmatic balance:
        - annual_loss_rate_pct legitimately reaches ~0.52 because historical
          loss genuinely predicts future loss (that's the whole point).
        - The old leaked exg_change_2018_2022 had r=-0.184 — too low to
          catch here. The ExG leak is caught by the column-name test and
          the feature manifest instead.
        - A feature with |r| > 0.55 is suspicious enough to investigate.
          It won't catch all leaks, but it catches severe ones.
        """
        from scipy.stats import pearsonr

        feature_cols = [c for c in FEATURE_COLUMNS_FULL if c in palawan_df.columns]
        y = palawan_df["high_risk"].values

        for col in feature_cols:
            x = palawan_df[col].values
            if np.std(x) < 1e-10:
                continue  # skip constant features
            r, _ = pearsonr(x, y)
            assert abs(r) < 0.55, (
                f"Feature '{col}' has suspicious correlation with target: r={r:.3f}. "
                f"This may indicate temporal leakage."
            )

    def test_exg_feature_is_pre_target(self, palawan_df):
        """ExG change feature must use only pre-target imagery (2018->2019)."""
        # The feature name encodes the years used
        exg_cols = [c for c in palawan_df.columns if c.startswith("exg_change")]
        for col in exg_cols:
            # Extract years from column name
            parts = col.replace("exg_change_", "").split("_")
            if len(parts) == 2:
                year_after = int(parts[1])
                assert year_after < 2020, (
                    f"ExG feature '{col}' uses {year_after} imagery — this is in "
                    f"the target period (2020-2022) and constitutes temporal leakage."
                )

    def test_neighbor_features_use_pre_period_loss(self, palawan_df):
        """Neighbor loss features must correlate with pre_period_loss, not high_risk.

        If neighbor_loss_count were computed from high_risk instead of
        pre_period_loss, it would correlate more strongly with high_risk
        than with pre_period_loss.
        """
        if "neighbor_loss_count" not in palawan_df.columns:
            pytest.skip("neighbor_loss_count not computed")

        from scipy.stats import pearsonr

        nlc = palawan_df["neighbor_loss_count"].values
        hr = palawan_df["high_risk"].values
        ppl = palawan_df["pre_period_loss"].values

        r_hr, _ = pearsonr(nlc, hr)
        r_ppl, _ = pearsonr(nlc, ppl)

        # If computed from pre_period_loss, correlation with pre_period_loss
        # should be stronger than with high_risk
        assert r_ppl > r_hr, (
            f"neighbor_loss_count correlates more with high_risk (r={r_hr:.3f}) "
            f"than pre_period_loss (r={r_ppl:.3f}) — likely leaking target"
        )


class TestModelQuality:
    """Basic model quality checks — not excellence gates, but sanity checks."""

    def test_model_trains_and_produces_valid_metrics(self):
        """Model should train and produce valid metric values."""
        from core.ml.training import train_model
        from core.ml.evaluation import evaluate_model

        df = load_dataset(region="palawan")
        X, y, feature_names = prepare_features(df)

        from core.ml.dataset import split_data
        X_train, X_test, y_train, y_test = split_data(X, y)

        model = train_model(X_train, y_train, n_estimators=50, max_depth=8)
        metrics = evaluate_model(model, X_test, y_test, feature_names)

        # Metrics should be real numbers in valid ranges
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1

        # Model should predict both classes (not degenerate)
        preds = model.predict(X_test)
        assert len(set(preds)) == 2, "Model only predicts one class — degenerate"

    def test_cross_region_generalization(self):
        """Train on 2 regions, test on held-out region.

        This is a smoke test, not a quality gate. It verifies that the model
        produces predictions on unseen regions and that the prediction
        distribution is not degenerate (all-zero or all-one). It does NOT
        assert good F1 — with 0.7% positive rate in Sierra Madre, meaningful
        F1 on a held-out region is not expected from this baseline.
        """
        from core.ml.training import train_model

        # Train on Palawan + Mindanao, test on Sierra Madre
        train_df = pd.concat([
            load_dataset(region="palawan"),
            load_dataset(region="mindanao_bukidnon"),
        ], ignore_index=True)
        test_df = load_dataset(region="sierra_madre")

        common_features = [
            c for c in FEATURE_COLUMNS_FULL
            if c in train_df.columns and c in test_df.columns
        ]

        X_train = train_df[common_features].values.astype(np.float64)
        y_train = train_df["high_risk"].values
        X_test = test_df[common_features].values.astype(np.float64)
        y_test = test_df["high_risk"].values

        model = train_model(X_train, y_train, n_estimators=50, max_depth=8)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        # Must produce valid predictions
        assert len(preds) == len(y_test)
        assert set(preds).issubset({0, 1})

        # Probabilities must span a range (not degenerate)
        assert probs.max() > probs.min(), "Model produces constant probabilities — degenerate"

        # Must predict at least some positive cells (not all-zero)
        # With class_weight=balanced the model should flag some cells
        assert preds.sum() > 0, (
            "Cross-region model predicts zero high-risk cells — "
            "no signal transferred between regions"
        )
