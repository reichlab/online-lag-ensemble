"""
Predictor functions
"""

import xarray as xr
from truth import mask_truths
from ledge.datatypes import Prediction, Truth
from ledge.utils import uniform_weights
from utils.dists import weighted_prediction
from functools import partial
from typing import List
from inspect import signature


def _predictor(truths: List[Truth], preds: List[Prediction], loss_fn, fill_fn,
               merging_fn, update_fn):
    output = []
    epiweeks = preds[0].coords["epiweek"]

    # Extract init_weights from update_fn
    init_weights = signature(update_fn).parameters["init_weights"].default
    weights_history = []

    for idx, ew in enumerate(epiweeks):
        if idx == 0:
            if init_weights is None:
                weights = uniform_weights([pred.attrs["model"] for pred in preds], ones=False)
            else:
                weights = init_weights
        else:
            truth = merging_fn(fill_fn(mask_truths(truths, ew)))
            losses = [loss_fn(pred.loc[:(ew - 1)], truth) for pred in preds]
            weights = update_fn(losses)
            weights = weights / weights.sum()

        weights_history.append(weights)

        output.append(weighted_prediction([pred.loc[ew] for pred in preds], weights))

    return xr.concat(output, dim="epiweek"), weights_history


def make_predictor(loss_fn, merging_fn, update_fn, fill_fn=lambda x: x):
    return partial(_predictor, loss_fn=loss_fn, fill_fn=fill_fn,
                   merging_fn=merging_fn, update_fn=update_fn)
