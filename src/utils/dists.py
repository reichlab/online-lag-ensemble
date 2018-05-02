"""
Utilities for working with distributions
"""

import numpy as np
import xarray as xr
from ledge.datatypes import Truth, Prediction, Weight
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


def actual_to_one_hot(vector: np.ndarray, target: str, multibin=False) -> np.ndarray:
    """
    Actual values to one hot encoded bins
    """

    bins = BINS[target]

    y = np.zeros((vector.shape[0], bins.shape[0]))

    indices = np.digitize(vector, bins, right=False) - 1

    expand = np.zeros_like(vector, dtype=int)
    if multibin:
        if target in ["onset-wk", "peak-wk"]:
            # Week bins. We expand 1 neighbour on each side
            expand[:] = 1
            if target == "onset-wk":
                # Don't expand the last bin (index = 34) which signifies 'none'
                # onset
                expand[indices == 34] = 0
        else:
            # These are percent bins, we expand 5 neighbours on the sides
            expand[:] = 5

    for i in range(vector.shape[0]):
        # Expand indices
        expanded_indices = []
        for e in range(-expand[i], expand[i] + 1):
            idx = indices[i] + e
            if idx >= 0 and idx < len(bins):
                expanded_indices.append(idx)

        y[i, expanded_indices] = 1

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


def probabilities(pred: Prediction, truth: Truth, multibin=False) -> xr.DataArray:
    """
    Return probabilities for the prediction.
    """

    one_hot = actual_to_one_hot(truth, pred.pattrs["target"], multibin=multibin)
    return np.multiply(one_hot, pred).sum(axis=1)
