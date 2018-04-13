"""
Utilities for working with distributions
"""

import numpy as np
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


def weighted_ensemble(dists: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """
    Return weighted ensemble
    """

    return np.sum([d * w for d, w in zip(dists, weights)], axis=0)


def mean_ensemble(dists: List[np.ndarray]) -> np.ndarray:
    """
    Return mean of dists. Works as mean ensemble model.
    """

    return weighted_ensemble(dists, np.ones(len(dists)) / len(dists))


def prediction_probabilities(Xs: List[np.ndarray], y: np.ndarray, target: str) -> np.ndarray:
    """
    Return score matrix for the predictions
    """

    return np.stack([np.multiply(actual_to_one_hot(y, target), X).sum(axis=1) for X in Xs], axis=1)
