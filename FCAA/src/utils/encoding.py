"""
Solution encoding and decoding for the combined feature selection
and hyperparameter tuning problem.

Each solution vector x ∈ R^D encodes:
  - Feature mask: first N_feat dims, thresholded at 0.5 → binary keep/drop
  - Hyperparameters: remaining dims, linearly mapped from [0,1] to [param_min, param_max]
"""

from typing import Dict, List, Tuple

import numpy as np


class SolutionEncoding:
    """
    Handles encoding/decoding of solution vectors for the FS+HPO problem.

    Parameters
    ----------
    n_features : int
        Total number of features in the dataset.
    hyperparameter_bounds : dict
        Mapping of hyperparameter name → (min, max, scale).
        scale can be 'linear' or 'log'.
    """

    def __init__(
        self,
        n_features: int,
        hyperparameter_bounds: Dict[str, Tuple[float, float, str]],
    ):
        self.n_features = n_features
        self.hyperparameter_bounds = hyperparameter_bounds
        self.hp_names = list(hyperparameter_bounds.keys())
        self.n_hyperparams = len(self.hp_names)
        self.total_dimension = n_features + self.n_hyperparams

    def decode(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float], np.ndarray]:
        """
        Decode a solution vector into feature mask and hyperparameters.

        Parameters
        ----------
        x : np.ndarray, shape (total_dimension,)
            The solution vector in [0, 1] space.

        Returns
        -------
        feature_mask : np.ndarray, shape (n_features,), dtype bool
            Binary mask: True = keep feature.
        hyperparameters : dict
            Decoded hyperparameter values in their natural ranges.
        raw_indices : np.ndarray, shape (n_selected,)
            Indices of selected features.
        """
        # Feature selection: threshold at 0.5
        raw_features = x[: self.n_features]
        feature_mask = raw_features > 0.5
        raw_indices = np.where(feature_mask)[0]

        # Hyperparameter mapping
        hyperparameters = {}
        for i, name in enumerate(self.hp_names):
            raw_val = x[self.n_features + i]
            vmin, vmax, scale = self.hyperparameter_bounds[name]
            raw_val = np.clip(raw_val, 0.0, 1.0)
            if scale == "log":
                # Map [0, 1] → [vmin, vmax] on log scale
                log_val = np.log10(vmin) + raw_val * (np.log10(vmax) - np.log10(vmin))
                hyperparameters[name] = float(10 ** log_val)
            elif scale == "integer":
                hyperparameters[name] = int(np.round(vmin + raw_val * (vmax - vmin)))
            else:  # 'linear'
                hyperparameters[name] = float(vmin + raw_val * (vmax - vmin))

        return feature_mask, hyperparameters, raw_indices

    def encode_to_vector(
        self, feature_mask: np.ndarray, hyperparameters: Dict[str, float]
    ) -> np.ndarray:
        """
        Encode a feature mask and hyperparameters into a solution vector.

        Parameters
        ----------
        feature_mask : np.ndarray, shape (n_features,), dtype bool
        hyperparameters : dict
            Hyperparameter values in natural ranges.

        Returns
        -------
        x : np.ndarray, shape (total_dimension,)
        """
        x = np.zeros(self.total_dimension)
        # Features: convert bool to continuous (0.2 for False, 0.8 for True)
        x[: self.n_features] = np.where(feature_mask, 0.8, 0.2)

        # Hyperparameters: reverse map to [0, 1]
        for i, name in enumerate(self.hp_names):
            val = hyperparameters[name]
            vmin, vmax, scale = self.hyperparameter_bounds[name]
            if scale == "log":
                raw = (np.log10(val) - np.log10(vmin)) / (
                    np.log10(vmax) - np.log10(vmin)
                )
            elif scale == "integer":
                raw = (val - vmin) / (vmax - vmin)
            else:
                raw = (val - vmin) / (vmax - vmin)
            x[self.n_features + i] = np.clip(raw, 0.0, 1.0)

        return x

    def random_population(self, pop_size: int, seed: int = None) -> np.ndarray:
        """
        Generate a uniformly random initial population.

        Parameters
        ----------
        pop_size : int
            Number of individuals.
        seed : int, optional
            Random seed.

        Returns
        -------
        population : np.ndarray, shape (pop_size, total_dimension)
        """
        rng = np.random.default_rng(seed)
        return rng.uniform(0.0, 1.0, size=(pop_size, self.total_dimension))

    def get_dimension_info(self) -> Dict:
        """Return a summary of the encoding dimensions."""
        return {
            "n_features": self.n_features,
            "n_hyperparams": self.n_hyperparams,
            "total_dimension": self.total_dimension,
            "hp_names": self.hp_names,
            "hp_bounds": self.hyperparameter_bounds,
        }
