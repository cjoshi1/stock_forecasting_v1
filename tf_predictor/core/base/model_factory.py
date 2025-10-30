"""
Factory for creating time series prediction models.

This module implements the Factory Pattern to enable pluggable model architectures.
"""

from typing import Dict, Any, Optional, Type
import torch.nn as nn

from .model_interface import TimeSeriesModel


def _register_builtin_models():
    """Register built-in models with the factory."""
    try:
        from ..ft_model import FTTransformerCLSModel
        ModelFactory.register_model('ft_transformer_cls', FTTransformerCLSModel)
    except ImportError:
        pass  # FT-Transformer not available

    try:
        from ..csn_model import CSNTransformerCLSModel
        ModelFactory.register_model('csn_transformer_cls', CSNTransformerCLSModel)
    except ImportError:
        pass  # CSN-Transformer not available


class ModelFactory:
    """
    Factory class for creating time series prediction models.

    This factory enables:
    1. Dynamic model creation based on string identifiers
    2. Easy registration of new model types
    3. Centralized model instantiation logic
    4. Type-safe model creation

    Usage:
        # Register a model
        ModelFactory.register_model('my_model', MyModelClass)

        # Create a model instance
        model = ModelFactory.create_model(
            model_type='ft_transformer',
            sequence_length=10,
            num_features=5,
            output_dim=1,
            d_model=64
        )
    """

    # Registry of available models
    _model_registry: Dict[str, Type[TimeSeriesModel]] = {}

    @classmethod
    def register_model(cls, model_type: str, model_class: Type[TimeSeriesModel]) -> None:
        """
        Register a new model type in the factory.

        Args:
            model_type: String identifier for the model (e.g., 'ft_transformer')
            model_class: Class that implements TimeSeriesModel interface

        Raises:
            ValueError: If model_type is already registered
            TypeError: If model_class doesn't inherit from TimeSeriesModel
        """
        if not issubclass(model_class, TimeSeriesModel):
            raise TypeError(
                f"Model class {model_class.__name__} must inherit from TimeSeriesModel"
            )

        if model_type in cls._model_registry:
            raise ValueError(
                f"Model type '{model_type}' is already registered. "
                f"Use unregister_model() first if you want to replace it."
            )

        cls._model_registry[model_type] = model_class

    @classmethod
    def unregister_model(cls, model_type: str) -> None:
        """
        Unregister a model type from the factory.

        Args:
            model_type: String identifier for the model to remove
        """
        if model_type in cls._model_registry:
            del cls._model_registry[model_type]

    @classmethod
    def create_model(
        cls,
        model_type: str,
        sequence_length: int,
        output_dim: int,
        num_features: Optional[int] = None,
        num_numerical: Optional[int] = None,
        num_categorical: Optional[int] = None,
        cat_cardinalities: Optional[list] = None,
        **model_kwargs
    ) -> TimeSeriesModel:
        """
        Create a model instance.

        Args:
            model_type: String identifier for the model type
            sequence_length: Length of input sequences
            output_dim: Output dimension (num_targets * prediction_horizon)
            num_features: Number of input features per time step (for non-CLS models)
            num_numerical: Number of numerical features (for CLS models)
            num_categorical: Number of categorical features (for CLS models)
            cat_cardinalities: Cardinalities for categorical features (for CLS models)
            **model_kwargs: Additional model-specific parameters

        Returns:
            Instance of the requested model

        Raises:
            ValueError: If model_type is not registered

        Example:
            # Standard model
            model = ModelFactory.create_model(
                model_type='ft_transformer',
                sequence_length=10,
                num_features=5,
                output_dim=2,
                d_model=64,
                num_heads=4,
                num_layers=3
            )

            # CLS model with categorical features
            model = ModelFactory.create_model(
                model_type='ft_transformer_cls',
                sequence_length=10,
                num_numerical=8,
                num_categorical=2,
                cat_cardinalities=[100, 5],
                output_dim=2,
                d_model=128,
                num_heads=8,
                num_layers=3
            )
        """
        if model_type not in cls._model_registry:
            available = list(cls._model_registry.keys())
            raise ValueError(
                f"Unknown model type: '{model_type}'. "
                f"Available models: {available}. "
                f"Use ModelFactory.register_model() to add new models."
            )

        model_class = cls._model_registry[model_type]

        # Create model instance with appropriate parameters
        # CLS models use different signature (num_numerical, num_categorical, cat_cardinalities)
        if model_type.endswith('_cls'):
            if num_numerical is None or num_categorical is None:
                raise ValueError(
                    f"CLS models require 'num_numerical' and 'num_categorical' parameters. "
                    f"Got num_numerical={num_numerical}, num_categorical={num_categorical}"
                )
            model = model_class(
                sequence_length=sequence_length,
                num_numerical=num_numerical,
                num_categorical=num_categorical,
                cat_cardinalities=cat_cardinalities or [],
                output_dim=output_dim,
                **model_kwargs
            )
        else:
            # Standard models use num_features parameter
            if num_features is None:
                raise ValueError(
                    f"Non-CLS models require 'num_features' parameter. "
                    f"Got num_features={num_features}"
                )
            model = model_class(
                sequence_length=sequence_length,
                num_features=num_features,
                output_dim=output_dim,
                **model_kwargs
            )

        return model

    @classmethod
    def get_available_models(cls) -> list:
        """
        Get list of available model types.

        Returns:
            List of registered model type identifiers
        """
        return list(cls._model_registry.keys())

    @classmethod
    def is_model_registered(cls, model_type: str) -> bool:
        """
        Check if a model type is registered.

        Args:
            model_type: String identifier to check

        Returns:
            True if model is registered, False otherwise
        """
        return model_type in cls._model_registry

    @classmethod
    def get_model_class(cls, model_type: str) -> Type[TimeSeriesModel]:
        """
        Get the class for a registered model type.

        Args:
            model_type: String identifier for the model

        Returns:
            The model class

        Raises:
            ValueError: If model_type is not registered
        """
        if model_type not in cls._model_registry:
            available = list(cls._model_registry.keys())
            raise ValueError(
                f"Unknown model type: '{model_type}'. "
                f"Available models: {available}"
            )

        return cls._model_registry[model_type]


def get_default_model_params(model_type: str) -> Dict[str, Any]:
    """
    Get default parameters for a model type.

    Args:
        model_type: String identifier for the model

    Returns:
        Dictionary of default parameters

    Example:
        params = get_default_model_params('ft_transformer')
        # Returns: {'d_model': 64, 'num_heads': 4, 'num_layers': 3, ...}
    """
    defaults = {
        'ft_transformer_cls': {
            'd_model': 128,
            'num_heads': 8,
            'num_layers': 3,
            'dropout': 0.1,
            'activation': 'gelu'
        },
        'csn_transformer_cls': {
            'd_model': 128,
            'num_heads': 8,
            'num_layers': 3,
            'dropout': 0.1,
            'activation': 'gelu'
        },
        'lstm': {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'bidirectional': False
        },
        'gru': {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'bidirectional': False
        }
    }

    return defaults.get(model_type, {})


# Auto-register built-in models when module is imported
_register_builtin_models()
