
import itertools
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_inspect
_ADJOINTS = {}
_CHOLESKY_DECOMPS = {}
_MATMUL = {}
_SOLVE = {}
_INVERSES = {}
def _registered_function(type_list, registry):
  enumerated_hierarchies = [enumerate(tf_inspect.getmro(t)) for t in type_list]
  cls_combinations = list(itertools.product(*enumerated_hierarchies))
  def hierarchy_distance(cls_combination):
    candidate_distance = sum(c[0] for c in cls_combination)
    if tuple(c[1] for c in cls_combination) in registry:
      return candidate_distance
    return 10000
  registered_combination = min(cls_combinations, key=hierarchy_distance)
  return registry.get(tuple(r[1] for r in registered_combination), None)
def _registered_adjoint(type_a):
  return _registered_function([type_a], _ADJOINTS)
def _registered_cholesky(type_a):
  return _registered_function([type_a], _CHOLESKY_DECOMPS)
def _registered_matmul(type_a, type_b):
  return _registered_function([type_a, type_b], _MATMUL)
def _registered_solve(type_a, type_b):
  return _registered_function([type_a, type_b], _SOLVE)
def _registered_inverse(type_a):
  return _registered_function([type_a], _INVERSES)
def adjoint(lin_op_a, name=None):
  adjoint_fn = _registered_adjoint(type(lin_op_a))
  if adjoint_fn is None:
    raise ValueError("No adjoint registered for {}".format(
        type(lin_op_a)))
  with ops.name_scope(name, "Adjoint"):
    return adjoint_fn(lin_op_a)
def cholesky(lin_op_a, name=None):
  cholesky_fn = _registered_cholesky(type(lin_op_a))
  if cholesky_fn is None:
    raise ValueError("No cholesky decomposition registered for {}".format(
        type(lin_op_a)))
  with ops.name_scope(name, "Cholesky"):
    return cholesky_fn(lin_op_a)
def matmul(lin_op_a, lin_op_b, name=None):
  """Compute lin_op_a.matmul(lin_op_b).
  Args:
    lin_op_a: The LinearOperator on the left.
    lin_op_b: The LinearOperator on the right.
    name: Name to use for this operation.
  Returns:
    A LinearOperator that represents the matmul between `lin_op_a` and
      `lin_op_b`.
  Raises:
    NotImplementedError: If no matmul method is defined between types of
      `lin_op_a` and `lin_op_b`.
  """
  matmul_fn = _registered_matmul(type(lin_op_a), type(lin_op_b))
  if matmul_fn is None:
    raise ValueError("No matmul registered for {}.matmul({})".format(
        type(lin_op_a), type(lin_op_b)))
  with ops.name_scope(name, "Matmul"):
    return matmul_fn(lin_op_a, lin_op_b)
def solve(lin_op_a, lin_op_b, name=None):
  """Compute lin_op_a.solve(lin_op_b).
  Args:
    lin_op_a: The LinearOperator on the left.
    lin_op_b: The LinearOperator on the right.
    name: Name to use for this operation.
  Returns:
    A LinearOperator that represents the solve between `lin_op_a` and
      `lin_op_b`.
  Raises:
    NotImplementedError: If no solve method is defined between types of
      `lin_op_a` and `lin_op_b`.
  """
  solve_fn = _registered_solve(type(lin_op_a), type(lin_op_b))
  if solve_fn is None:
    raise ValueError("No solve registered for {}.solve({})".format(
        type(lin_op_a), type(lin_op_b)))
  with ops.name_scope(name, "Solve"):
    return solve_fn(lin_op_a, lin_op_b)
def inverse(lin_op_a, name=None):
  inverse_fn = _registered_inverse(type(lin_op_a))
  if inverse_fn is None:
    raise ValueError("No inverse registered for {}".format(
        type(lin_op_a)))
  with ops.name_scope(name, "Inverse"):
    return inverse_fn(lin_op_a)
class RegisterAdjoint:
  """Decorator to register an Adjoint implementation function.
  Usage:
  @linear_operator_algebra.RegisterAdjoint(lin_op.LinearOperatorIdentity)
  def _adjoint_identity(lin_op_a):
  """
  def __init__(self, lin_op_cls_a):
    self._key = (lin_op_cls_a,)
  def __call__(self, adjoint_fn):
    if not callable(adjoint_fn):
      raise TypeError(
          "adjoint_fn must be callable, received: {}".format(adjoint_fn))
    if self._key in _ADJOINTS:
      raise ValueError("Adjoint({}) has already been registered to: {}".format(
          self._key[0].__name__, _ADJOINTS[self._key]))
    _ADJOINTS[self._key] = adjoint_fn
    return adjoint_fn
class RegisterCholesky:
  """Decorator to register a Cholesky implementation function.
  Usage:
  @linear_operator_algebra.RegisterCholesky(lin_op.LinearOperatorIdentity)
  def _cholesky_identity(lin_op_a):
  """
  def __init__(self, lin_op_cls_a):
    self._key = (lin_op_cls_a,)
  def __call__(self, cholesky_fn):
    if not callable(cholesky_fn):
      raise TypeError(
          "cholesky_fn must be callable, received: {}".format(cholesky_fn))
    if self._key in _CHOLESKY_DECOMPS:
      raise ValueError("Cholesky({}) has already been registered to: {}".format(
          self._key[0].__name__, _CHOLESKY_DECOMPS[self._key]))
    _CHOLESKY_DECOMPS[self._key] = cholesky_fn
    return cholesky_fn
class RegisterMatmul:
  """Decorator to register a Matmul implementation function.
  Usage:
  @linear_operator_algebra.RegisterMatmul(
    lin_op.LinearOperatorIdentity,
    lin_op.LinearOperatorIdentity)
  def _matmul_identity(a, b):
  """
  def __init__(self, lin_op_cls_a, lin_op_cls_b):
    self._key = (lin_op_cls_a, lin_op_cls_b)
  def __call__(self, matmul_fn):
    if not callable(matmul_fn):
      raise TypeError(
          "matmul_fn must be callable, received: {}".format(matmul_fn))
    if self._key in _MATMUL:
      raise ValueError("Matmul({}, {}) has already been registered.".format(
          self._key[0].__name__,
          self._key[1].__name__))
    _MATMUL[self._key] = matmul_fn
    return matmul_fn
class RegisterSolve:
  """Decorator to register a Solve implementation function.
  Usage:
  @linear_operator_algebra.RegisterSolve(
    lin_op.LinearOperatorIdentity,
    lin_op.LinearOperatorIdentity)
  def _solve_identity(a, b):
  """
  def __init__(self, lin_op_cls_a, lin_op_cls_b):
    self._key = (lin_op_cls_a, lin_op_cls_b)
  def __call__(self, solve_fn):
    if not callable(solve_fn):
      raise TypeError(
          "solve_fn must be callable, received: {}".format(solve_fn))
    if self._key in _SOLVE:
      raise ValueError("Solve({}, {}) has already been registered.".format(
          self._key[0].__name__,
          self._key[1].__name__))
    _SOLVE[self._key] = solve_fn
    return solve_fn
class RegisterInverse:
  """Decorator to register an Inverse implementation function.
  Usage:
  @linear_operator_algebra.RegisterInverse(lin_op.LinearOperatorIdentity)
  def _inverse_identity(lin_op_a):
  """
  def __init__(self, lin_op_cls_a):
    self._key = (lin_op_cls_a,)
  def __call__(self, inverse_fn):
    if not callable(inverse_fn):
      raise TypeError(
          "inverse_fn must be callable, received: {}".format(inverse_fn))
    if self._key in _INVERSES:
      raise ValueError("Inverse({}) has already been registered to: {}".format(
          self._key[0].__name__, _INVERSES[self._key]))
    _INVERSES[self._key] = inverse_fn
    return inverse_fn
