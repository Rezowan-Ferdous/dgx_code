"""
Registry for models and datasets
Allows dynamic registration and retrieval of components
"""

from typing import Dict, Callable, Any, Optional


class Registry:
    """
    A registry to map strings to classes or functions

    This allows for dynamic instantiation of models, datasets, etc.
    """

    def __init__(self, name: str):
        """
        Initialize registry

        Args:
            name: Name of the registry
        """
        self._name = name
        self._registry: Dict[str, Callable] = {}

    def register(self, name: Optional[str] = None):
        """
        Decorator to register a class or function

        Args:
            name: Name to register (uses class/function name if None)

        Example:
            @MODEL_REGISTRY.register()
            class MyModel(nn.Module):
                pass

            @MODEL_REGISTRY.register("custom_name")
            class AnotherModel(nn.Module):
                pass
        """
        def decorator(obj: Callable) -> Callable:
            obj_name = name if name is not None else obj.__name__
            if obj_name in self._registry:
                raise KeyError(
                    f"{obj_name} is already registered in {self._name}"
                )
            self._registry[obj_name] = obj
            return obj

        return decorator

    def get(self, name: str) -> Callable:
        """
        Get registered object by name

        Args:
            name: Name of registered object

        Returns:
            Registered object

        Raises:
            KeyError: If name is not registered
        """
        if name not in self._registry:
            raise KeyError(
                f"{name} is not registered in {self._name}. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def list(self) -> list:
        """List all registered names"""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        """Check if name is registered"""
        return name in self._registry

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name}, " \
               f"registered={list(self._registry.keys())})"


# Global registries
MODEL_REGISTRY = Registry("models")
DATASET_REGISTRY = Registry("datasets")
LOSS_REGISTRY = Registry("losses")
OPTIMIZER_REGISTRY = Registry("optimizers")


def build_from_registry(registry: Registry, name: str, **kwargs) -> Any:
    """
    Build object from registry

    Args:
        registry: Registry to use
        name: Name of registered object
        **kwargs: Arguments to pass to constructor

    Returns:
        Instantiated object
    """
    obj_class = registry.get(name)
    return obj_class(**kwargs)
