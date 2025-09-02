"""
ScalarLM-specific interfaces and protocols for vLLM integration.
"""

from typing import ClassVar, Literal, Protocol, Type, Union, overload, runtime_checkable
from typing_extensions import TypeIs

import torch


@runtime_checkable
class SupportsTokenformer(Protocol):
    """The interface required for all models that support Tokenformer."""
    
    supports_tokenformer: ClassVar[Literal[True]] = True
    """
    A flag that indicates this model supports Tokenformer operations.
    
    Note:
        There is no need to redefine this flag if this class is in the
        MRO of your model class.
    """
    
    def enable_tokenformer(self, config: dict) -> None:
        """Enable Tokenformer functionality for this model."""
        ...
    
    def disable_tokenformer(self) -> None:
        """Disable Tokenformer functionality for this model."""
        ...


# We can't use runtime_checkable with ClassVar for issubclass checks
# so we need to treat the class as an instance and use isinstance instead
@runtime_checkable
class _SupportsTokenformerType(Protocol):
    supports_tokenformer: Literal[True]


@overload
def supports_tokenformer(model: Type[object]) -> TypeIs[Type[SupportsTokenformer]]:
    ...


@overload 
def supports_tokenformer(model: object) -> TypeIs[SupportsTokenformer]:
    ...


def supports_tokenformer(
    model: Union[Type[object], object],
) -> Union[TypeIs[Type[SupportsTokenformer]], TypeIs[SupportsTokenformer]]:
    if isinstance(model, type):
        return isinstance(model(), _SupportsTokenformerType)
    else:
        return isinstance(model, _SupportsTokenformerType)