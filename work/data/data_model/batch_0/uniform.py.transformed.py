
import math
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=["distributions.Uniform"])
class Uniform(distribution.Distribution):
  """Uniform distribution with `low` and `high` parameters.
  The probability density function (pdf) is,
  ```none
  pdf(x; a, b) = I[a <= x < b] / Z
  Z = b - a
  ```
  where
  - `low = a`,
  - `high = b`,
  - `Z` is the normalizing constant, and
  - `I[predicate]` is the [indicator function](
    https://en.wikipedia.org/wiki/Indicator_function) for `predicate`.
  The parameters `low` and `high` must be shaped in a way that supports
  broadcasting (e.g., `high - low` is a valid operation).
  ```python
  u2 = Uniform(low=[1.0, 2.0],
  u3 = Uniform(low=[[1.0, 2.0],
                    [3.0, 4.0]],
               high=[[1.5, 2.5],
  ```
  ```python
  ```
  """
  @deprecation.deprecated(
      "2019-01-01",
      "The TensorFlow Distributions library has moved to "
      "TensorFlow Probability "
      "(https://github.com/tensorflow/probability). You "
      "should update all references to use `tfp.distributions` "
      "instead of `tf.distributions`.",
      warn_once=True)
  def __init__(self,
               low=0.,
               high=1.,
               validate_args=False,
               allow_nan_stats=True,
               name="Uniform"):
    """Initialize a batch of Uniform distributions.
    Args:
      low: Floating point tensor, lower boundary of the output interval. Must
        have `low < high`.
      high: Floating point tensor, upper boundary of the output interval. Must
        have `low < high`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    Raises:
      InvalidArgumentError: if `low >= high` and `validate_args=False`.
    """
    parameters = dict(locals())
    with ops.name_scope(name, values=[low, high]) as name:
      with ops.control_dependencies([
          check_ops.assert_less(
              low, high, message="uniform not defined when low >= high.")
      ] if validate_args else []):
        self._low = array_ops.identity(low, name="low")
        self._high = array_ops.identity(high, name="high")
        check_ops.assert_same_float_dtype([self._low, self._high])
    super(Uniform, self).__init__(
        dtype=self._low.dtype,
        reparameterization_type=distribution.FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._low,
                       self._high],
        name=name)
  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(("low", "high"),
            ([ops.convert_to_tensor(sample_shape, dtype=dtypes.int32)] * 2)))
  @property
  def low(self):
    return self._low
  @property
  def high(self):
    return self._high
  def range(self, name="range"):
    with self._name_scope(name):
      return self.high - self.low
  def _batch_shape_tensor(self):
    return array_ops.broadcast_dynamic_shape(
        array_ops.shape(self.low),
        array_ops.shape(self.high))
  def _batch_shape(self):
    return array_ops.broadcast_static_shape(
        self.low.get_shape(),
        self.high.get_shape())
  def _event_shape_tensor(self):
    return constant_op.constant([], dtype=dtypes.int32)
  def _event_shape(self):
    return tensor_shape.TensorShape([])
  def _sample_n(self, n, seed=None):
    shape = array_ops.concat([[n], self.batch_shape_tensor()], 0)
    samples = random_ops.random_uniform(shape=shape,
                                        dtype=self.dtype,
                                        seed=seed)
    return self.low + self.range() * samples
  def _prob(self, x):
    broadcasted_x = x * array_ops.ones(
        self.batch_shape_tensor(), dtype=x.dtype)
    return array_ops.where_v2(
        math_ops.is_nan(broadcasted_x), broadcasted_x,
        array_ops.where_v2(
            math_ops.logical_or(broadcasted_x < self.low,
                                broadcasted_x >= self.high),
            array_ops.zeros_like(broadcasted_x),
            array_ops.ones_like(broadcasted_x) / self.range()))
  def _cdf(self, x):
    broadcast_shape = array_ops.broadcast_dynamic_shape(
        array_ops.shape(x), self.batch_shape_tensor())
    zeros = array_ops.zeros(broadcast_shape, dtype=self.dtype)
    ones = array_ops.ones(broadcast_shape, dtype=self.dtype)
    broadcasted_x = x * ones
    result_if_not_big = array_ops.where_v2(
        x < self.low, zeros, (broadcasted_x - self.low) / self.range())
    return array_ops.where_v2(x >= self.high, ones, result_if_not_big)
  def _entropy(self):
    return math_ops.log(self.range())
  def _mean(self):
    return (self.low + self.high) / 2.
  def _variance(self):
    return math_ops.square(self.range()) / 12.
  def _stddev(self):
    return self.range() / math.sqrt(12.)
