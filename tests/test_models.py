"""Tests for model definitions, Pipelines, and non-negative predictions."""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.features import build_model_pipeline
from src.models import build_model_zoo, get_model_specifications, predict_features


def test_model_zoo_contains_required_models_and_uses_pipelines() -> None:
    """All requested regressors, including Poisson loss, are available."""
    features = pd.DataFrame({"numeric": [0.0, 1.0], "category": ["a", "b"]})
    model_zoo = build_model_zoo(features)

    assert set(model_zoo) == set(get_model_specifications())
    assert len(model_zoo) == 9
    assert all(isinstance(model, Pipeline) for model in model_zoo.values())
    assert (
        model_zoo["HistGradientBoosting"].named_steps["model"].early_stopping is False
    )
    assert (
        model_zoo["HistGradientBoosting Poisson"].named_steps["model"].loss == "poisson"
    )


def test_build_model_zoo_rejects_unknown_names() -> None:
    """Misspelled model names fail before training begins."""
    with pytest.raises(ValueError, match="Unknown model names"):
        build_model_zoo(pd.DataFrame({"value": [1]}), ["not a model"])


def test_predict_clips_negative_outputs() -> None:
    """The shared prediction helper enforces the non-negative count range."""
    train_features = pd.DataFrame({"value": [0.0, 1.0]})
    pipeline = build_model_pipeline(Ridge(alpha=0), train_features, scale_numeric=True)
    pipeline.fit(train_features, [10.0, 0.0])
    predictions = predict_features(pipeline, pd.DataFrame({"value": [3.0]}))

    assert np.array_equal(predictions, [0.0])


def test_predict_requires_a_fitted_pipeline() -> None:
    """An unfitted model fails clearly rather than returning nonsense."""
    pipeline = build_model_pipeline(Ridge(), pd.DataFrame({"value": [1.0]}))
    with pytest.raises(NotFittedError):
        predict_features(pipeline, pd.DataFrame({"value": [2.0]}))
