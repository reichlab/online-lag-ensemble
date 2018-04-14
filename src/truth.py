from ledge.datatypes import Truth
from ledge.utils import get_lag
from pymmwr import Epiweek
from typing import List


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
