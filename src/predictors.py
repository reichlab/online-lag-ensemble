"""
Predictor functions
"""

import xarray as xr
from truth import mask_truths
from ledge.datatypes import Prediction, Truth
from utils.dists import weighted_prediction
from functools import partial
from typing import List


def _predictor(truths: List[Truth],
               preds: List[Prediction],
               loss_fn, merging_fn, update_fn,
               init_weights=None):
    output = []
    epiweeks = preds[0].coords["epiweek"]

    for idx, ew in enumerate(epiweeks):
        if idx == 0:
            weights = init_weights
        else:
            truth = merging_fn(mask_truths(truths, ew))
            losses = [loss_fn(pred.loc[:(ew - 1)], truth) for pred in preds]
            weights = update_fn(losses, init_weights=init_weights)
            weights /= weights.sum()

        output.append(weighted_prediction([pred.loc[ew] for pred in preds], weights))

    return xr.concat(output, dim="epiweek")


def make_predictor(loss_fn, merging_fn, update_fn):
    return partial(_predictor,
                   loss_fn=loss_fn,
                   merging_fn=merging_fn,
                   update_fn=update_fn)
