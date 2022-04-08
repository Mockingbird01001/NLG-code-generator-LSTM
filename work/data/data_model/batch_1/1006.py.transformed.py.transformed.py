
import operator
import numpy as np
from typing import Iterable
array = np.array([1, 2])
def ndarray_func(x):
    return x
ndarray_func(np.array([1, 2]))
array == 1
array.dtype == float
np.dtype(float)
np.dtype(np.float64)
np.dtype(None)
np.dtype("float64")
np.dtype(np.dtype(float))
np.dtype(("U", 10))
np.dtype((np.int32, (2, 2)))
two_tuples_dtype = [("R", "u1"), ("G", "u1"), ("B", "u1")]
np.dtype(two_tuples_dtype)
three_tuples_dtype = [("R", "u1", 2)]
np.dtype(three_tuples_dtype)
mixed_tuples_dtype = [("R", "u1"), ("G", np.unicode_, 1)]
np.dtype(mixed_tuples_dtype)
shape_tuple_dtype = [("R", "u1", (2, 2))]
np.dtype(shape_tuple_dtype)
shape_like_dtype = [("R", "u1", (2, 2)), ("G", np.unicode_, 1)]
np.dtype(shape_like_dtype)
object_dtype = [("field1", object)]
np.dtype(object_dtype)
np.dtype((np.int32, (np.int8, 4)))
np.dtype(float) == float
np.dtype(float) != np.float64
np.dtype(float) < None
np.dtype(float) <= "float64"
np.dtype(float) > np.dtype(float)
np.dtype(float) >= np.dtype(("U", 10))
def iterable_func(x):
    return x
iterable_func(array)
[element for element in array]
iter(array)
zip(array, array)
array[1]
array[:]
array[...]
array[:] = 0
array_2d = np.ones((3, 3))
array_2d[:2, :2]
array_2d[..., 0]
array_2d[:2, :2] = 0
len(array)
str(array)
array_scalar = np.array(1)
int(array_scalar)
float(array_scalar)
bytes(array_scalar)
operator.index(array_scalar)
bool(array_scalar)
array < 1
array <= 1
array == 1
array != 1
array > 1
array >= 1
1 < array
1 <= array
1 == array
1 != array
1 > array
1 >= array
array + 1
1 + array
array += 1
array - 1
1 - array
array -= 1
array * 1
1 * array
array *= 1
nonzero_array = np.array([1, 2])
array / 1
1 / nonzero_array
float_array = np.array([1.0, 2.0])
float_array /= 1
array // 1
1 // nonzero_array
array //= 1
array % 1
1 % nonzero_array
array %= 1
divmod(array, 1)
divmod(1, nonzero_array)
array ** 1
1 ** array
array **= 1
array << 1
1 << array
array <<= 1
array >> 1
1 >> array
array >>= 1
array & 1
1 & array
array &= 1
array ^ 1
1 ^ array
array ^= 1
array | 1
1 | array
array |= 1
-array
+array
abs(array)
~array
np.array([1, 2]).transpose()
