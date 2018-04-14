"""
Helper module for working with data
"""

import numpy as np
import pandas as pd
import xarray as xr
import os
import os.path as path
import pymmwr
from ledge.datatypes import Truth, Prediction
from functools import lru_cache
from typing import List, Tuple
from utils.misc import epiweek_to_model_week


# Types
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
    selection = selection & (index["region"] == region_name)
    first_ew = (season * 100) + 40

    if pymmwr.epiweeks_in_year(season) == 53:
        # We skip the 20yy20 data and only provide upto 20yy19
        # This makes it easy for the visualizations
        last_ew = ((season + 1) * 100) + 19
    else:
        last_ew = ((season + 1) * 100) + 20

    selection = selection & (index["epiweek"] >= first_ew) & (index["epiweek"] <= last_ew)

    return index[selection].reset_index(drop=True), data[selection]


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
    def get(self, target_name: str, region_name: str, season: int) -> Prediction:
        """
        Return data for asked target_name along with index

        Parameters
        ----------
        target_name : str
            Identifier for the target to load
        region_name : str
            Short region code (nat, hhs2 ...)
        season : int
            Season, as identified by its first year
        """

        data = np.loadtxt(path.join(self.model_path, target_name))
        index, data = _narrow_selection(self.index, data, region_name, season)

        # Convert to pymmwr type
        epiweeks = [pymmwr.Epiweek(year=ew // 100, week=ew % 100) for ew in index["epiweek"]]

        meta = { "model": self.name, "region": region_name, "target": target_name }
        return xr.DataArray(data, dims=("epiweek", "bins"), coords={ "epiweek": epiweeks }, attrs=meta)


class ActualData:
    """
    Data loader for actual data
    """

    def __init__(self, exp_dir: str) -> None:
        self.exp_dir = exp_dir
        self._df = pd.read_csv(path.join(exp_dir, "actual.csv"))

    def get(self, target_name: str, region_name: str, season: int, lag: int) -> Truth:
        """
        Return index and data for given region.

        Parameters
        ----------
        target_name : str
            Identifier for the target to provide
        region_name : str
            Short region code (nat, hhs2 ...)
        season : int
            Season, as identified by its first year
        lag : int
            Lag to return
        """

        lag_df = self._df[self._df["lag"] == lag]
        index = lag_df[["epiweek", "region"]]
        data = lag_df[target_name].values

        if target_name in ["peak-wk", "onset-wk"]:
            # We use a general model week specification
            data = np.array([epiweek_to_model_week(ew) for ew in data])

        index, data = _narrow_selection(index, data, region_name, season)

        # Convert to pymmwr type
        epiweeks = [pymmwr.Epiweek(year=ew // 100, week=ew % 100) for ew in index["epiweek"]]

        meta = { "lag": lag, "region": region_name, "target": target_name }
        return xr.DataArray(data, dims="epiweek", coords={ "epiweek": epiweeks }, attrs=meta)


def available_models(exp_dir: str) -> List[str]:
    """
    Return name of models available as components in exp_dir
    """

    return sorted([
        model for model in
        os.listdir(exp_dir)
        if path.isdir(path.join(exp_dir, model))
    ])
