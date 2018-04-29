from ledge.datatypes import Truth
from ledge.utils import get_lag
import ledge.fill as fill
from pymmwr import Epiweek
from typing import List, Dict
from functools import partial
from hyperopt import hp


def mask_truths(truths: List[Truth], ew: Epiweek) -> List[Truth]:
    """
    Mask truths to return state as seen at epiweek ew (not including ew)
    """

    masked = []
    for truth in truths:
        end = ew - get_lag(truth) - 1
        clipped = truth.loc[:end]
        if len(clipped) > 0:
            masked.append(clipped)

    return masked


FILL_SPACE = hp.choice("imputation_type", [
    { "type": None },
    {
        "type": "diff",
        "lookback": 1 + hp.randint("lookback", 33),
        "normalize": hp.choice("normalize", [True, False]),
        "window": hp.choice("window", [
            {
                "type": "uniform"
            },
            {
                "type": "linear",
                "alpha": hp.uniform("alpha", 0, 1)
            },
            {
                "type": "geometric",
                "gamma": hp.uniform("gamma", 0, 1)
            }
        ]),
        "incremental": hp.choice("incremental", [True, False]),
        "n_series": 2 + hp.randint("n_series", 30)
    }
])


def impute(truths: List[Truth], config: Dict) -> List[Truth]:
    """
    Impute the list of truths according to the config
    """

    if (len(truths) < 2) or (config["type"] is None):
        return truths
    elif config["type"] is "diff":
        window = config["window"]
        if window["type"] == "uniform":
            window_fn = fill.window_uniform
        elif window["type"] == "linear":
            window_fn = partial(fill.window_linear, alpha=window["alpha"])
        elif window["type"] == "geometric":
            window_fn = partial(fill.window_geometric, gamma=window["gamma"])
        else:
            raise Exception("Window type not understood")

        window_fn = fill.lookback(config["lookback"])(window_fn)
        if config["normalize"]:
            window_fn = fill.normalize(window_fn)

        truths[:config["n_series"]] = fill.diff_window(
            truths[:config["n_series"]],
            window_fn,
            inc=config["incremental"]
        )
        return truths
    else:
        raise Exception("Imputation type not understood")
