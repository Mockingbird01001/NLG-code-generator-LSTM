import sys
from typing import Any, List, Sequence, Tuple, Union, TYPE_CHECKING
from numpy import dtype
from ._shape import _ShapeLike
if sys.version_info >= (3, 8):
    from typing import Protocol, TypedDict
    HAVE_PROTOCOL = True
else:
    try:
        from typing_extensions import Protocol, TypedDict
    except ImportError:
        HAVE_PROTOCOL = False
    else:
        HAVE_PROTOCOL = True
_DTypeLikeNested = Any
if TYPE_CHECKING or HAVE_PROTOCOL:
    class _DTypeDictBase(TypedDict):
        names: Sequence[str]
        formats: Sequence[_DTypeLikeNested]
    class _DTypeDict(_DTypeDictBase, total=False):
        offsets: Sequence[int]
        titles: Sequence[Any]
        itemsize: int
        aligned: bool
    class _SupportsDType(Protocol):
        dtype: _DTypeLikeNested
else:
    _DTypeDict = Any
    _SupportsDType = Any
_VoidDTypeLike = Union[
    Tuple[_DTypeLikeNested, int],
    Tuple[_DTypeLikeNested, _ShapeLike],
    List[Any],
    _DTypeDict,
    Tuple[_DTypeLikeNested, _DTypeLikeNested],
]
DTypeLike = Union[
    dtype,
    None,
    type,
    _SupportsDType,
    str,
    _VoidDTypeLike,
]
