from typing import Union, Tuple, Any
import numpy as np
_CharLike = Union[str, bytes]
_BoolLike = Union[bool, np.bool_]
_IntLike = Union[int, np.integer]
_FloatLike = Union[_IntLike, float, np.floating]
_ComplexLike = Union[_FloatLike, complex, np.complexfloating]
_NumberLike = Union[int, float, complex, np.number, np.bool_]
_ScalarLike = Union[
    int,
    float,
    complex,
    str,
    bytes,
    np.generic,
]
_VoidLike = Union[Tuple[Any, ...], np.void]
