"""
Utilities for working with distributions and similar stuff
"""

import numpy as np
import os
import os.path as path
import pymmwr
import warnings


def epiweek_to_season(epiweek: int) -> int:
    """
    A proper season starts from 20xx30 and ends at 20yy29.
    """

    year, week = epiweek // 100, epiweek % 100

    return year if week >= 30 else year - 1


def epiweek_to_model_week(epiweek: int) -> int:
    """
    Convert epiweek to model week starting at 20xx40 and ending at
    20xx19/20xx20
    """

    if np.isnan(epiweek):
        # We consider this as the sign of no onset and assign the last bin
        return 34

    epiweek = int(epiweek)
    season = epiweek_to_season(epiweek)
    nweeks = pymmwr.mmwr_weeks_in_year(season)

    week = epiweek % 100
    if week >= 40 and week <= nweeks:
        return epiweek - ((season * 100) + 40)
    elif (nweeks == 52 and week <= 21) or (nweeks == 53 and week <= 20):
        return week + (nweeks - 40)
    elif week >= 20:
        warnings.warn(f"Epiweek {epiweek} is greater than the models' range")
        return week + (nweeks - 40)
    else:
        raise Exception(f"Epiweek {epiweek} outside the desired range")


def ensure_dir(directory: str) -> str:
    if not path.exists(directory):
        os.makedirs(directory)
    return directory
