from typing import Dict, Type

from custom_lstm.models.base_model import AblationModel


class ModelRegistry:
    """
    Registry for Ablation Study Models.
    Allows dynamic registration and instantiation of models.
    """

    _registry: Dict[str, Type[AblationModel]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a model class with the given name.
        """

        def wrapper(wrapped_class: Type[AblationModel]):
            if not issubclass(wrapped_class, AblationModel):
                raise ValueError(f"Model {wrapped_class.__name__} must inherit from AblationModel")
            cls._registry[name] = wrapped_class
            return wrapped_class

        return wrapper

    @classmethod
    def build(cls, name: str, **kwargs) -> AblationModel:
        """
        Builds and returns an instance of the requested model.
        """
        if name not in cls._registry:
            raise KeyError(f"Model '{name}' not found in registry. Available models: {list(cls._registry.keys())}")

        model_class = cls._registry[name]
        return model_class(**kwargs)
