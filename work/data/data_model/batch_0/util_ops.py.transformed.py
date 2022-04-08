
import sys
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
def gcd(a, b, name=None):
  with ops.name_scope(name, 'gcd', [a, b]):
    a = ops.convert_to_tensor(a)
    b = ops.convert_to_tensor(b)
    a.shape.assert_has_rank(0)
    b.shape.assert_has_rank(0)
    if not a.dtype.is_integer:
      raise ValueError('a must be an integer type. Got: %s' % a.dtype)
    if not b.dtype.is_integer:
      raise ValueError('b must be an integer type. Got: %s' % b.dtype)
    const_a = tensor_util.constant_value(a)
    const_b = tensor_util.constant_value(b)
    if const_a is not None and const_b is not None:
      if sys.version_info.major < 3:
        math_gcd = fractions.gcd
      else:
        math_gcd = math.gcd
      return ops.convert_to_tensor(math_gcd(const_a, const_b))
    cond = lambda _, b: math_ops.greater(b, array_ops.zeros_like(b))
    body = lambda a, b: [b, math_ops.mod(a, b)]
    a, b = control_flow_ops.while_loop(cond, body, [a, b], back_prop=False)
    return a
