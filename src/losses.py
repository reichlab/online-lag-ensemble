"""
Loss calculation
"""

import numpy as np
from ledge.datatypes import Truth, Loss, Prediction
from utils.dists import probabilities


EPSILON = np.sqrt(np.finfo(float).eps)


def _attrs_check(pred: Prediction, truth: Truth):
    if any(pred.attrs[f] != truth.attrs[f] for f in ["target", "region"]):
        raise Exception("Attributes don't match for prediction and truth")


def ploss(pred: Prediction, truth: Truth) -> Loss:
    """
    Return loss as given by 1 - probability of true bin
    """

    _attrs_check(pred, truth)
    loss = 1 - probabilities(pred, truth)
    # Reattach metadata
    loss.attrs = { **pred.attrs, **truth.attrs }
    return loss


def logloss(pred: Prediction, truth: Truth) -> Loss:
    """
    Return negative log of probability of true bin
    """

    _attrs_check(pred, truth)
    loss = -np.log(probabilities(pred, truth) + EPSILON)
    loss.attrs = { **pred.attrs, **truth.attrs }
    return loss


def absloss(pred: Prediction, truth: Truth) -> Loss:
    """
    Return absolute loss based on peak of probability
    """

    _attrs_check(pred, truth)
    raise NotImplementedError()
