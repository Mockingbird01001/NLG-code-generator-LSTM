
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
__all__ = [
    "Dirichlet",
]
_dirichlet_sample_note = """Note: `value` must be a non-negative tensor with
dtype `self.dtype` and be in the `(self.event_shape() - 1)`-simplex, i.e.,
`tf.reduce_sum(value, -1) = 1`. It must have a shape compatible with
`self.batch_shape() + self.event_shape()`."""
@tf_export(v1=["distributions.Dirichlet"])
class Dirichlet(distribution.Distribution):
  """Dirichlet distribution.
  The Dirichlet distribution is defined over the
  [`(k-1)`-simplex](https://en.wikipedia.org/wiki/Simplex) using a positive,
  length-`k` vector `concentration` (`k > 1`). The Dirichlet is identically the
  Beta distribution when `k = 2`.
  The Dirichlet is a distribution over the open `(k-1)`-simplex, i.e.,
  ```none
  S^{k-1} = { (x_0, ..., x_{k-1}) in R^k : sum_j x_j = 1 and all_j x_j > 0 }.
  ```
  The probability density function (pdf) is,
  ```none
  pdf(x; alpha) = prod_j x_j**(alpha_j - 1) / Z
  Z = prod_j Gamma(alpha_j) / Gamma(sum_j alpha_j)
  ```
  where:
  * `x in S^{k-1}`, i.e., the `(k-1)`-simplex,
  * `concentration = alpha = [alpha_0, ..., alpha_{k-1}]`, `alpha_j > 0`,
  * `Z` is the normalization constant aka the [multivariate beta function](
    and,
  * `Gamma` is the [gamma function](
    https://en.wikipedia.org/wiki/Gamma_function).
  The `concentration` represents mean total counts of class occurrence, i.e.,
  ```none
  concentration = alpha = mean * total_concentration
  ```
  where `mean` in `S^{k-1}` and `total_concentration` is a positive real number
  representing a mean total count.
  Distribution parameters are automatically broadcast in all functions; see
  examples for details.
  Warning: Some components of the samples can be zero due to finite precision.
  This happens more often when some of the concentrations are very small.
  Make sure to round the samples to `np.finfo(dtype).tiny` before computing the
  density.
  Samples of this distribution are reparameterized (pathwise differentiable).
  The derivatives are computed using the approach described in
  (Figurnov et al., 2018).
  ```python
  import tensorflow_probability as tfp
  tfd = tfp.distributions
  alpha = [1., 2, 3]
  dist = tfd.Dirichlet(alpha)
  x = [[.1, .4, .5],
       [.2, .3, .5]]
  ```
  ```python
  alpha = [[1., 2, 3],
  dist = tfd.Dirichlet(alpha)
  x = [.2, .3, .5]
  ```
  Compute the gradients of samples w.r.t. the parameters:
  ```python
  alpha = tf.constant([1.0, 2.0, 3.0])
  dist = tfd.Dirichlet(alpha)
  grads = tf.gradients(loss, alpha)
  ```
  References:
    Implicit Reparameterization Gradients:
      [Figurnov et al., 2018]
      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients)
      ([pdf]
      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients.pdf))
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
               concentration,
               validate_args=False,
               allow_nan_stats=True,
               name="Dirichlet"):
    """Initialize a batch of Dirichlet distributions.
    Args:
      concentration: Positive floating-point `Tensor` indicating mean number
        of class occurrences; aka "alpha". Implies `self.dtype`, and
        `self.batch_shape`, `self.event_shape`, i.e., if
        `concentration.shape = [N1, N2, ..., Nm, k]` then
        `batch_shape = [N1, N2, ..., Nm]` and
        `event_shape = [k]`.
      validate_args: Python `bool`, default `False`. When `True` distribution
        parameters are checked for validity despite possibly degrading runtime
        performance. When `False` invalid inputs may silently render incorrect
        outputs.
      allow_nan_stats: Python `bool`, default `True`. When `True`, statistics
        (e.g., mean, mode, variance) use the value "`NaN`" to indicate the
        result is undefined. When `False`, an exception is raised if one or
        more of the statistic's batch members are undefined.
      name: Python `str` name prefixed to Ops created by this class.
    """
    parameters = dict(locals())
    with ops.name_scope(name, values=[concentration]) as name:
      self._concentration = self._maybe_assert_valid_concentration(
          ops.convert_to_tensor(concentration, name="concentration"),
          validate_args)
      self._total_concentration = math_ops.reduce_sum(self._concentration, -1)
    super(Dirichlet, self).__init__(
        dtype=self._concentration.dtype,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        reparameterization_type=distribution.FULLY_REPARAMETERIZED,
        parameters=parameters,
        graph_parents=[self._concentration,
                       self._total_concentration],
        name=name)
  @property
  def concentration(self):
    return self._concentration
  @property
  def total_concentration(self):
    return self._total_concentration
  def _batch_shape_tensor(self):
    return array_ops.shape(self.total_concentration)
  def _batch_shape(self):
    return self.total_concentration.get_shape()
  def _event_shape_tensor(self):
    return array_ops.shape(self.concentration)[-1:]
  def _event_shape(self):
    return self.concentration.get_shape().with_rank_at_least(1)[-1:]
  def _sample_n(self, n, seed=None):
    gamma_sample = random_ops.random_gamma(
        shape=[n],
        alpha=self.concentration,
        dtype=self.dtype,
        seed=seed)
    return gamma_sample / math_ops.reduce_sum(gamma_sample, -1, keepdims=True)
  @distribution_util.AppendDocstring(_dirichlet_sample_note)
  def _log_prob(self, x):
    return self._log_unnormalized_prob(x) - self._log_normalization()
  @distribution_util.AppendDocstring(_dirichlet_sample_note)
  def _prob(self, x):
    return math_ops.exp(self._log_prob(x))
  def _log_unnormalized_prob(self, x):
    x = self._maybe_assert_valid_sample(x)
    return math_ops.reduce_sum(math_ops.xlogy(self.concentration - 1., x), -1)
  def _log_normalization(self):
    return special_math_ops.lbeta(self.concentration)
  def _entropy(self):
    k = math_ops.cast(self.event_shape_tensor()[0], self.dtype)
    return (
        self._log_normalization()
        + ((self.total_concentration - k)
           * math_ops.digamma(self.total_concentration))
        - math_ops.reduce_sum(
            (self.concentration - 1.) * math_ops.digamma(self.concentration),
            axis=-1))
  def _mean(self):
    return self.concentration / self.total_concentration[..., array_ops.newaxis]
  def _covariance(self):
    x = self._variance_scale_term() * self._mean()
    return array_ops.matrix_set_diag(
        -math_ops.matmul(
            x[..., array_ops.newaxis],
        self._variance())
  def _variance(self):
    scale = self._variance_scale_term()
    x = scale * self._mean()
    return x * (scale - x)
  def _variance_scale_term(self):
    return math_ops.rsqrt(1. + self.total_concentration[..., array_ops.newaxis])
  @distribution_util.AppendDocstring(
)
  def _mode(self):
    k = math_ops.cast(self.event_shape_tensor()[0], self.dtype)
    mode = (self.concentration - 1.) / (
        self.total_concentration[..., array_ops.newaxis] - k)
    if self.allow_nan_stats:
      nan = array_ops.fill(
          array_ops.shape(mode),
          np.array(np.nan, dtype=self.dtype.as_numpy_dtype()),
          name="nan")
      return array_ops.where_v2(
          math_ops.reduce_all(self.concentration > 1., axis=-1), mode, nan)
    return control_flow_ops.with_dependencies([
        check_ops.assert_less(
            array_ops.ones([], self.dtype),
            self.concentration,
            message="Mode undefined when any concentration <= 1"),
    ], mode)
  def _maybe_assert_valid_concentration(self, concentration, validate_args):
    if not validate_args:
      return concentration
    return control_flow_ops.with_dependencies([
        check_ops.assert_positive(
            concentration,
            message="Concentration parameter must be positive."),
        check_ops.assert_rank_at_least(
            concentration, 1,
            message="Concentration parameter must have >=1 dimensions."),
        check_ops.assert_less(
            1, array_ops.shape(concentration)[-1],
            message="Concentration parameter must have event_size >= 2."),
    ], concentration)
  def _maybe_assert_valid_sample(self, x):
    if not self.validate_args:
      return x
    return control_flow_ops.with_dependencies([
        check_ops.assert_positive(x, message="samples must be positive"),
        check_ops.assert_near(
            array_ops.ones([], dtype=self.dtype),
            math_ops.reduce_sum(x, -1),
            message="sample last-dimension must sum to `1`"),
    ], x)
@kullback_leibler.RegisterKL(Dirichlet, Dirichlet)
def _kl_dirichlet_dirichlet(d1, d2, name=None):
  """Batchwise KL divergence KL(d1 || d2) with d1 and d2 Dirichlet.
  Args:
    d1: instance of a Dirichlet distribution object.
    d2: instance of a Dirichlet distribution object.
    name: (optional) Name to use for created operations.
      default is "kl_dirichlet_dirichlet".
  Returns:
    Batchwise KL(d1 || d2)
  """
  with ops.name_scope(name, "kl_dirichlet_dirichlet", values=[
      d1.concentration, d2.concentration]):
    digamma_sum_d1 = math_ops.digamma(
        math_ops.reduce_sum(d1.concentration, axis=-1, keepdims=True))
    digamma_diff = math_ops.digamma(d1.concentration) - digamma_sum_d1
    concentration_diff = d1.concentration - d2.concentration
    return (math_ops.reduce_sum(concentration_diff * digamma_diff, axis=-1) -
            special_math_ops.lbeta(d1.concentration) +
            special_math_ops.lbeta(d2.concentration))
