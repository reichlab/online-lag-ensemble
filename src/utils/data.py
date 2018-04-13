"""
Helper module for working with data
"""

import numpy as np
import pandas as pd
import os
import os.path as path
import pymmwr
from functools import lru_cache
from typing import List, Tuple
from utils.misc import epiweek_to_model_week


Index = pd.DataFrame
Data = np.ndarray


def _narrow_selection(index: Index, data: Data, region_name: str, season: int) -> Tuple[Index, Data]:
    """
    Return a narrowed index and data using the region and season information on the index.
    Require index to have epiweek and region
    """

    assert index.shape[0] == data.shape[0], "Shape of index and data not matching in first dimension"

    # All true selection
    selection = index["epiweek"] > 0
    narrowing = False
    if region_name is not None:
        selection = selection & (index["region"] == region_name)
        narrowing = True

    if season is not None:
        first_ew = (season * 100) + 40

        if pymmwr.mmwr_weeks_in_year(season) == 53:
            # We skip the 20yy20 data and only provide upto 20yy19
            # This makes it easy for the visualizations
            last_ew = ((season + 1) * 100) + 19
        else:
            last_ew = ((season + 1) * 100) + 20

        selection = selection & (index["epiweek"] >= first_ew) & (index["epiweek"] <= last_ew)
        narrowing = True

    if narrowing:
        return index[selection].reset_index(drop=True), data[selection]
    else:
        return index, data


class Component:
    """
    Loader for component models
    """

    def __init__(self, exp_dir: str, name: str) -> None:
        self.name = name
        self.exp_dir = exp_dir
        self.model_path = path.join(exp_dir, name)
        self.index = pd.read_csv(path.join(exp_dir, "index.csv"))

    @lru_cache(None)
    def get(self, target_name: str, region_name=None, season=None) -> Tuple[Index, Data]:
        """
        Return data for asked target_name along with index

        Parameters
        ----------
        target_name : str
            Identifier for the target to load
        region_name : str | None
            Short region code (nat, hhs2 ...) or None for all regions
        season : int | None
            Season, as identified by its first year, or None for all seasons
        """

        data = np.loadtxt(path.join(self.model_path, target_name))
        return _narrow_selection(self.index, data, region_name, season)


class ActualData:
    """
    Data loader for actual data
    """

    def __init__(self, exp_dir: str) -> None:
        self.exp_dir = exp_dir
        self._df = pd.read_csv(path.join(exp_dir, "actual.csv"))

    def get(self, target_name: str, region_name=None, season=None, latest=True) -> Tuple[Index, Data]:
        """
        Return index and data for given region.

        Parameters
        ----------
        target_name : str
            Identifier for the target to provide
        region_name : str | None
            Short region code (nat, hhs2 ...) or None for all regions
        season : int | None
            Season, as identified by its first year, or None for all seasons
        latest : bool
            Whether to return the latest truth or the first available one
        """

        target_column = f"{target_name}-{'latest' if latest else 'first'}"
        index = self._df[["epiweek", "region"]]
        data = self._df[target_column].values

        if target_name in ["peak-wk", "onset-wk"]:
            # We use a general model week specification
            data = np.array([epiweek_to_model_week(ew) for ew in data])

        return _narrow_selection(index, data, region_name, season)


def available_models(exp_dir: str) -> List[str]:
    """
    Return name of models available as components in exp_dir
    """

    return sorted([
        model for model in
        os.listdir(exp_dir)
        if path.isdir(path.join(exp_dir, model))
    ])


def get_seasons_data(ad: ActualData, cmps: List[Component], seasons: List[int], target: str, region: str, latest=True) -> Tuple[Index, List[Data], Data]:
    """
    Return a tuple of yi, Xs and y for the givens seasons, concatenated.
    """

    ypairs = [ad.get(target, region_name=region, season=s, latest=latest) for s in seasons]
    yi = pd.concat([yp[0] for yp in ypairs], ignore_index=True)
    Xs = [
        np.concatenate([c.get(target, region_name=region, season=s)[1] for s in seasons])
        for c in cmps
    ]
    y = np.concatenate([yp[1] for yp in ypairs])

    return yi, Xs, y
