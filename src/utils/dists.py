"""
Utilities for working with distributions
"""

import numpy as np
import xarray as xr
from ledge.datatypes import Truth, Prediction, Weight
from ledge.utils import uniform_weights
from typing import List


BINS = {
    "1-ahead": np.linspace(0, 13, 131),
    "2-ahead": np.linspace(0, 13, 131),
    "3-ahead": np.linspace(0, 13, 131),
    "4-ahead": np.linspace(0, 13, 131),
    "peak": np.linspace(0, 13, 131),
    "peak-wk": np.arange(0, 34),
    "onset-wk": np.arange(0, 35)
}


def actual_to_one_hot(vector: np.ndarray, target: str) -> np.ndarray:
    """
    Actual values to one hot encoded bins
    """

    bins = BINS[target]

    y = np.zeros((vector.shape[0], bins.shape[0]))

    indices = np.digitize(vector, bins, right=False) - 1

    for i in range(vector.shape[0]):
        y[i, indices[i]] = 1

    return y


def weighted_prediction(preds: List[Prediction], weights: Weight) -> Prediction:
    merged = xr.merge([p.rename(p.attrs["model"]) for p in preds], join="left")
    merged = merged.to_array().rename({ "variable": "model" })
    merged = merged.dot(weights)

    # Reattach metadata
    for key in ["target", "region"]:
        if key in preds[0].attrs:
            merged.attrs[key] = preds[0].attrs[key]
    return merged


def mean_prediction(preds: List[Prediction]) -> Prediction:
    models = [pred.attrs["model"] for pred in preds]
    return weighted_prediction(preds, uniform_weights(models, ones=False))


def probabilities(pred: Prediction, truth: Truth) -> xr.DataArray:
    """
    Return probabilities for the prediction
    """

    return np.multiply(actual_to_one_hot(truth, pred.attrs["target"]), pred).sum(axis=1)
