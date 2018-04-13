"""
Models
"""

from abc import ABC, abstractmethod
from collections import Counter
from functools import lru_cache
from typing import List, Tuple
import json
import utils.dists as udists
import utils.misc as u
import numpy as np
import pandas as pd


EPSILON = np.sqrt(np.finfo(float).eps)


def beta_softmax(vector, beta):
    """
    Calculate softmax with a beta (inverse temperature parameter)
    """

    expv = np.exp(beta * vector)
    return expv / np.sum(expv)


def dem(mat, weights=None, epsilon=None):
    """
    Run the degenerate EM algorithm on given data. Return a set of weights for
    each model. Code replicates the method implemented in epiforecast-R package
    here https://github.com/cmu-delphi/epiforecast-R/blob/master/epiforecast/R/ensemble.R

    Parameters
    ----------
    mat : np.ndarray
        Shape (n_obs, n_models). Probabilities from n_models models for n_obs
        observations.
    weights : np.ndarray
        Initial weights
    epsilon : float
        Tolerance value
    """

    if weights is None:
        weights = np.ones(mat.shape[1]) / mat.shape[1]

    if not epsilon:
        epsilon = EPSILON

    w_mat = mat * weights
    marginals = np.sum(w_mat, axis=1)
    log_marginal = np.mean(np.log(marginals))

    if np.isneginf(log_marginal):
        raise ValueError("All methods assigned a probability of 0 to at least one observed event.")
    else:
        while True:
            prev_log_marginal = log_marginal
            weights = np.mean(w_mat.T / marginals, axis=1)
            w_mat = mat * weights
            marginals = np.sum(w_mat, axis=1)
            log_marginal = np.mean(np.log(marginals))

            if log_marginal + epsilon < prev_log_marginal:
                raise ValueError("Log marginal less than prev_log_marginal")
            marginal_diff = log_marginal - prev_log_marginal
            if (marginal_diff <= epsilon) or ((marginal_diff / -log_marginal) <= epsilon):
                break
    return weights


class Model(ABC):
    """
    A general time series model. Training is batched while prediction is not.
    """

    @property
    def params(self):
        return {
            "target": self.target,
            "n_comps": self.n_comps
        }

    @params.setter
    def params(self, ps):
        self.target = ps["target"]
        self.n_comps = ps["n_comps"]

    @property
    def fit_params(self):
        return {}

    @fit_params.setter
    def fit_params(self, fps):
        pass

    @abstractmethod
    def fit(self, index_vec, component_predictions_vec, truth_vec):
        """
        Train the model

        Parameters
        ----------
        index_vec : pd.DataFrame
            Dataframe with atleast two columns "epiweek" and "region".
            Shape is (n_instances, 2).
        component_predictions_vec : List[np.ndarray]
            One matrix for each component. Matrix is (n_instances, n_bins).
        truth_vec : np.ndarray
            True values. Shape (n_instances, )
        """
        ...

    @abstractmethod
    def predict(self, index, component_predictions, truth=None):
        """
        Predict output for a single timepoint

        Parameters
        ----------
        index : List/Tuple
            Pair of epiweek and region values
        component_predictions : List[np.ndarray]
            One vector for each component. Vector is of shape (n_bins, )
        truth
            True value for the time point. This is only used by oracular models.
        """
        ...

    def feedback(self, last_truth):
        """
        Take feedback in the form of last truth. This can then be used for updating the
        weights.
        """

        raise NotImplementedError("feedback not available in this model")


class SerializerMixin:
    """
    Mixin for serializing and deserializing a model
    """

    def save(self, file_name, with_state=False):
        """
        Save model data in given file. Optionally save state too.
        """

        with open(file_name, "w") as fp:
            data = {
                "params": self.params,
                "fit_params": self.fit_params
            }

            if with_state:
                data["state"] = self.state
            json.dump(data, fp)

    def load(self, file_name, with_state=False):
        """
        Load model data from given file. Optionally load state too.
        """

        with open(file_name) as fp:
            data = json.load(fp)

            self.params = data["params"]
            self.fit_params = data["fit_params"]
            if with_state:
                self.state = data["state"]


class OracleEnsemble(SerializerMixin, Model):
    """
    Oracular ensemble. Outputs the prediction from the best model.
    """

    def __init__(self, target: str, n_comps: int):
        self.target = target
        self.n_comps = n_comps

    def fit(self, index_vec, component_predictions_vec, truth_vec):
        pass

    def predict(self, index, component_predictions, truth):
        """
        Use the truth to identify the best component. Then output its
        prediction
        """

        one_hot = udists.actual_to_one_hot(np.array([truth]), self.target)
        best_model_idx = np.argmax((np.array(component_predictions) * one_hot).sum(axis=1))
        return component_predictions[best_model_idx]


class IthExpertEnsemble(SerializerMixin, Model):
    """
    Picks expert i and goes with it
    """

    def __init__(self, target: str, n_comps: int, i: int):
        self.target = target
        self.n_comps = n_comps
        self._i = i

    def fit(self, index_vec, component_predictions_vec, truth_vec):
        pass

    @property
    def params(self):
        return { "i": self._i, **super().params }

    @params.setter
    def params(self, params):
        Model.params.fset(self, params)
        self._i = params["i"]

    def predict(self, index, component_predictions):
        """
        Use the truth to identify the best component. Then output its
        prediction
        """

        return component_predictions[self._i]


class MeanEnsemble(SerializerMixin, Model):
    """
    Mean ensemble. Outputs the mean of predictions from the components.
    """

    def __init__(self, target: str, n_comps: int):
        self.target = target
        self.n_comps = n_comps

    def fit(self, index_vec, component_predictions_vec, truth_vec):
        pass

    def predict(self, index, component_predictions):
        """
        Use the truth to identify the best component. Then output its
        prediction
        """

        return udists.mean_ensemble(component_predictions)


class DemWeightEnsemble(SerializerMixin, Model):
    """
    Degenerate EM ensemble.
    """

    def __init__(self, target: str, n_comps: int):
        self.target = target
        self.n_comps = n_comps

    def fit(self, index_vec, component_predictions_vec, truth_vec):
        """
        Use degenerate EM to find the best set of weights optimizing the log scores
        """

        probabilities = udists.prediction_probabilities(component_predictions_vec, truth_vec, self.target)
        self._weights = dem(probabilities)
        score = np.log((probabilities * self._weights).sum(axis=1)).mean()

    @property
    def fit_params(self):
        return { "weights": self._weights.tolist(), **super().fit_params }

    @fit_params.setter
    def fit_params(self, fit_params):
        Model.fit_params.fset(self, fit_params)
        self._weights = np.array(fit_params["weights"])

    def predict(self, index, component_predictions):
        """
        Use the truth to identify the best component. Then output its
        prediction
        """

        return udists.weighted_ensemble(component_predictions, self._weights)


class HitWeightEnsemble(SerializerMixin, Model):
    """
    Ensemble that weighs components according to the number of times they have
    been the best. This is similar to the score weight ensemble but since hits
    are relatively sparse (by definition), we make this a whole training data
    thing as compared to the score weight model which is per week.
    """

    def __init__(self, target: str, n_comps: int, beta: float):
        """
        Parameters
        ----------
        target : str
            Target identifier
        n_comps : int
            Number of components
        beta : float
            Beta for the softmax
        """

        self.target = target
        self.n_comps = n_comps
        self._beta = beta

    @property
    def params(self):
        return { "beta": self._beta, **super().params }

    @params.setter
    def params(self, params):
        Model.params.fset(self, params)
        self._beta = params["beta"]

    @property
    def fit_params(self):
        return { "weights": self._weights.tolist(), **super().fit_params }

    @fit_params.setter
    def fit_params(self, fit_params):
        Model.fit_params.fset(self, fit_params)
        self._weights = np.array(fit_params["weights"])

    def fit(self, index_vec, component_predictions_vec, truth_vec):
        """
        Count the number of best hits and pass through the softmax to get weights
        """

        probabilities = udists.prediction_probabilities(component_predictions_vec, truth_vec, self.target)
        hits = Counter(np.argmax(probabilities, axis=1))
        self._weights = beta_softmax(np.array([hits[i] for i in range(self.n_comps)]), self._beta)

    def predict(self, index, component_predictions):
        """
        Use the truth to identify the best component. Then output its
        prediction
        """

        return udists.weighted_ensemble(component_predictions, self._weights)


class ScoreWeightEnsemble(SerializerMixin, Model):
    """
    Ensemble that weighs components according to the scores they get for each model week.
    """

    def __init__(self, target: str, n_comps: int, beta: float):
        """
        Parameters
        ----------
        target : str
            Target identifier
        n_comps : int
            Number of components
        beta : float
            Beta for the softmax
        """

        self.target = target
        self.n_comps = n_comps
        self._beta = beta

    @property
    def params(self):
        return { "beta": self._beta, **super().params }

    @params.setter
    def params(self, params):
        Model.params.fset(self, params)
        self._beta = params["beta"]

    @property
    def fit_params(self):
        return {
            "weights": self._weights.tolist(),
            "model_weeks": self._model_weeks,
            **super().fit_params
        }

    @fit_params.setter
    def fit_params(self, fit_params):
        Model.fit_params.fset(self, fit_params)
        self._weights = np.array(fit_params["weights"])
        self._model_weeks = np.array(fit_params["model_weeks"])

    def fit(self, index_vec, component_predictions_vec, truth_vec):
        """
        Group the scores according to model weeks
        """

        model_weeks = index_vec["epiweek"].map(u.epiweek_to_model_week)
        probabilities = udists.prediction_probabilities(component_predictions_vec, truth_vec, self.target)

        dfdict = { "model_weeks": model_weeks }
        for i in range(self.n_comps):
            dfdict[i] = probabilities[:, i]

        # Mean probabilities per model week
        mean_probabilities = pd.DataFrame(dfdict).groupby("model_weeks").mean()
        self._model_weeks = list(mean_probabilities.index)

        # beta softmax simplifies since log and exp cancel
        probs = mean_probabilities.values
        probs **= self._beta
        self._weights = probs / probs.sum(axis=1)[:, None]

    def predict(self, index, component_predictions):
        """
        Use the truth to identify the best component. Then output its
        prediction
        """

        model_week = u.epiweek_to_model_week(index["epiweek"])
        weights = self._weights[list(self._model_weeks).index(model_week)]

        return udists.weighted_ensemble(component_predictions, weights)


class KDemWeightEnsemble(SerializerMixin, Model):
    """
    Degenerate EM ensemble trained on optimal k partition of epiweeks.
    """

    def __init__(self, target: str, n_comps: int, k: int):
        self.target = target
        self.n_comps = n_comps
        self._k = k

    @property
    def params(self):
        return { "k": self._k, **super().params }

    @params.setter
    def params(self, params):
        Model.params.fset(self, params)
        self._k = params["k"]

    @property
    def fit_params(self):
        return {
            "partition_lengths": self._partition_lengths,
            "partition_weights": np.array(self._partition_weights).tolist(),
            **super().fit_params
        }

    @fit_params.setter
    def fit_params(self, fit_params):
        Model.fit_params.fset(self, fit_params)
        self._partition_lengths = fit_params["partition_lengths"]
        self._partition_weights = np.array(fit_params["partition_weights"])

    @lru_cache(None)
    def _score_partition(self, start_wk, length):
        """
        For a partition specified by start_wk and length, fit weights
        and find score
        """

        # Model weeks in the partition
        weeks = list(range(start_wk, start_wk + length))
        selection = self._model_week_index.isin(weeks)

        weights = dem(self._probabilities[selection])
        score = np.log((self._probabilities[selection] * weights).sum(axis=1)).mean()

        return score, weights

    @lru_cache(None)
    def _partition(self, start_wk, k):
        """
        Find optimal number of partitions
        """

        if k == 1:
            # We work on the complete remaining chunk
            length = self._nweeks - start_wk
            score, weights = self._score_partition(start_wk, length)
            return score, [length], [weights]

        optimal_score = -np.inf
        optimal_lengths = []
        optimal_weights = []

        for length in range(1, (self._nweeks - start_wk) - (k - 1)):
            score, weights = self._score_partition(start_wk, length)
            rest_score, rest_lengths, rest_weights = self._partition(start_wk + length, k - 1)

            # Find the mean of scores
            remaining_length = self._nweeks - start_wk - length
            total_score = ((score * length) + (rest_score * remaining_length)) / (length + remaining_length)

            if total_score > optimal_score:
                optimal_score = total_score
                optimal_lengths = [length, *rest_lengths]
                optimal_weights = [weights, *rest_weights]

        return optimal_score, optimal_lengths, optimal_weights

    def fit(self, index_vec, component_predictions_vec, truth_vec):
        """
        Use degenerate EM to find the best set of weights optimizing the log scores.
        Assume model weeks is a sequence from 0 to nweeks - 1 with no gaps or other stuff
        """

        self._partition.cache_clear()
        self._score_partition.cache_clear()

        self._probabilities = udists.prediction_probabilities(component_predictions_vec, truth_vec, self.target)
        self._model_week_index = index_vec["epiweek"].map(u.epiweek_to_model_week)

        self._nweeks = len(np.unique(self._model_week_index))
        score, self._partition_lengths, self._partition_weights = self._partition(0, self._k)

    def predict(self, index, component_predictions):
        """
        Use the truth to identify the best component. Then output its
        prediction
        """

        model_week = u.epiweek_to_model_week(index["epiweek"])
        partitions = np.cumsum(self._partition_lengths)
        partition_idx = np.sum(partitions <= model_week)
        weights = self._partition_weights[partition_idx]

        return udists.weighted_ensemble(component_predictions, weights)


class MPWeightEnsemble(SerializerMixin, Model):
    """
    Simple multiplicative weighing algorithm (hedge).
    """

    def __init__(self, target: str, n_comps: int, beta: int):
        self.target = target
        self.n_comps = n_comps
        self._beta = beta
        self._past_predictions = []
        self._past_gains = []

    @property
    def params(self):
        return { "beta": self._beta, **super().params }

    @params.setter
    def params(self, params):
        Model.params.fset(self, params)
        self._beta = params["beta"]

    @property
    def state(self):
        return {
            "past_predictions": self._past_predictions,
            "past_gains": self._past_gains,
            "weights": self._weights
        }

    @state.setter
    def state(self, state):
        self._weights = state["weights"]
        self._past_gains = state["past_gains"]
        self._past_predictions = state["past_predictions"]

    def fit(self, index_vec, component_predictions_vec, truth_vec):
        self._weights = np.ones((self.n_comps,)) / self.n_comps
        self._past_predictions = []
        self._past_gains = []

    def predict(self, index, component_predictions):
        """
        Return prediction using current weights
        """

        self._past_predictions.append(component_predictions)
        return udists.weighted_ensemble(component_predictions, self._weights / np.sum(self._weights))

    def feedback(self, last_truth):
        """
        Use the truth from the last timepoint to update weights
        """

        last_probabilities = udists.prediction_probabilities(self._past_predictions[-1], np.array([last_truth]), self.target)[0]
        self._past_gains.append(last_probabilities)
        self._weights *= self._beta ** (1 - last_probabilities)
