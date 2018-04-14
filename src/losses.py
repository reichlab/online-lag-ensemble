"""
Loss calculation
"""

import xarray as xr
from ledge.datatypes import Truth, Loss, Prediction
from utils.dists import prediction_probabilities


def _attrs_check(pred: Prediction, truth: Truth):
    if any(pred.attrs[f] != truth.attrs[f] for f in ["target", "region"]):
        raise Exception("Attributes don't match for prediction and truth")


def ploss(pred: Prediction, truth: Truth) -> Loss:
    """
    Return loss as given by 1 - probability of true bin
    """

    _attrs_check(pred, truth)
    raise NotImplementedError()


def logloss(pred: Prediction, truth: Truth) -> Loss:
    """
    Return negative log of probability of true bin
    """

    _attrs_check(pred, truth)
    raise NotImplementedError()


def absloss(pred: Prediction, truth: Truth) -> Loss:
    """
    Return absolute loss based on peak of probability
    """

    _attrs_check(pred, truth)
    raise NotImplementedError()
