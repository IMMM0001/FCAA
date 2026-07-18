"""
Machine learning model wrappers for regression.

Each wrapper provides:
- hyperparameter_bounds(): returns {param_name: (min, max, scale)}
- build_model(**hparams): returns a sklearn-compatible model
- Unified interface for SVR, KRR, RandomForest, and MLP
"""

from typing import Dict, Tuple

from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


class ModelWrapper:
    """Base model wrapper with hyperparameter bounds."""

    @staticmethod
    def hyperparameter_bounds() -> Dict[str, Tuple[float, float, str]]:
        """
        Return hyperparameter bounds.

        Returns
        -------
        dict : {param_name: (min, max, scale)}
            scale is 'linear', 'log', or 'integer'.
        """
        raise NotImplementedError

    @staticmethod
    def build_model(**hparams) -> object:
        """Build a sklearn-compatible regressor with given hyperparameters."""
        raise NotImplementedError


class SVRWrapper(ModelWrapper):
    """
    Support Vector Regression (SVR) with RBF kernel.

    Hyperparameters:
    - C: regularization [1e-2, 1e4], log scale
    - epsilon: tube width [1e-3, 1.0], log scale
    - gamma: RBF kernel width [1e-4, 1e1], log scale
    """

    @staticmethod
    def hyperparameter_bounds() -> Dict[str, Tuple[float, float, str]]:
        return {
            "C": (1e-2, 1e4, "log"),
            "epsilon": (1e-3, 1.0, "log"),
            "gamma": (1e-4, 1e1, "log"),
        }

    @staticmethod
    def build_model(**hparams) -> SVR:
        return SVR(
            C=hparams.get("C", 1.0),
            epsilon=hparams.get("epsilon", 0.1),
            gamma=hparams.get("gamma", "scale"),
            kernel="rbf",
        )


class KRRWrapper(ModelWrapper):
    """
    Kernel Ridge Regression (KRR).

    Hyperparameters:
    - alpha: L2 regularization [1e-6, 1e1], log scale
    - gamma: RBF kernel width [1e-4, 1e1], log scale
    """

    @staticmethod
    def hyperparameter_bounds() -> Dict[str, Tuple[float, float, str]]:
        return {
            "alpha": (1e-6, 1e1, "log"),
            "gamma": (1e-4, 1e1, "log"),
        }

    @staticmethod
    def build_model(**hparams) -> KernelRidge:
        return KernelRidge(
            alpha=hparams.get("alpha", 1.0),
            gamma=hparams.get("gamma", None),
            kernel="rbf",
        )


class RandomForestWrapper(ModelWrapper):
    """
    Random Forest Regressor.

    Hyperparameters:
    - n_estimators: number of trees [10, 500], integer
    - max_depth: max tree depth [2, 30], integer
    - min_samples_split: [2, 20], integer
    - max_features: fraction of features [0.1, 1.0], linear
    """

    @staticmethod
    def hyperparameter_bounds() -> Dict[str, Tuple[float, float, str]]:
        return {
            "n_estimators": (10, 500, "integer"),
            "max_depth": (2, 30, "integer"),
            "min_samples_split": (2, 20, "integer"),
            "max_features": (0.1, 1.0, "linear"),
        }

    @staticmethod
    def build_model(**hparams) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=int(hparams.get("n_estimators", 100)),
            max_depth=int(hparams.get("max_depth", 10)),
            min_samples_split=int(hparams.get("min_samples_split", 2)),
            max_features=hparams.get("max_features", 1.0),
            random_state=42,
            n_jobs=-1,
        )


class MLPWrapper(ModelWrapper):
    """
    Multi-Layer Perceptron (MLP) Regressor.

    Hyperparameters:
    - hidden_size: neurons per layer [16, 256], integer
    - alpha: L2 regularization [1e-6, 1e1], log scale
    - learning_rate_init: initial learning rate [1e-5, 1e-1], log scale
    """

    @staticmethod
    def hyperparameter_bounds() -> Dict[str, Tuple[float, float, str]]:
        return {
            "hidden_size": (16, 256, "integer"),
            "alpha": (1e-6, 1e1, "log"),
            "learning_rate_init": (1e-5, 1e-1, "log"),
        }

    @staticmethod
    def build_model(**hparams) -> MLPRegressor:
        hidden_size = int(hparams.get("hidden_size", 100))
        return MLPRegressor(
            hidden_layer_sizes=(hidden_size, hidden_size // 2),
            alpha=hparams.get("alpha", 0.0001),
            learning_rate_init=hparams.get("learning_rate_init", 0.001),
            activation="relu",
            solver="adam",
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )


# Registry for easy model selection
MODEL_REGISTRY: Dict[str, type] = {
    "svr": SVRWrapper,
    "krr": KRRWrapper,
    "random_forest": RandomForestWrapper,
    "mlp": MLPWrapper,
}


def get_model_wrapper(model_name: str) -> ModelWrapper:
    """Get a model wrapper by name."""
    model_name = model_name.lower().replace(" ", "_")
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[model_name]
