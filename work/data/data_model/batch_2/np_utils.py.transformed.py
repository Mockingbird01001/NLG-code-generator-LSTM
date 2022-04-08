
import inspect
import numbers
import os
import re
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_export
from tensorflow.python.types import core
from tensorflow.python.util import nest
def _canonicalize_axis(axis, rank):
  return _canonicalize_axes([axis], rank)[0]
def _canonicalize_axes(axes, rank):
  rank = _maybe_static(rank)
  if isinstance(rank, core.Tensor):
    canonicalizer = (
        lambda axis: cond(axis < 0, lambda: axis + rank, lambda: axis))
  else:
    canonicalizer = lambda axis: axis + rank if axis < 0 else axis
  return [canonicalizer(axis) for axis in axes]
def _supports_signature():
  return hasattr(inspect, 'signature')
def _to_tf_type(dtype):
  return dtypes.as_dtype(dtype)
def _to_numpy_type(dtype):
  if isinstance(dtype, dtypes.DType):
    return dtype.as_numpy_dtype
  return np.dtype(dtype)
def isscalar(val):
  if isinstance(val, np_arrays.ndarray):
    val = val.data
  if isinstance(val, core.Tensor):
    ndims = val.shape.ndims
    if ndims is not None:
      return ndims == 0
    else:
      return math_ops.equal(array_ops.rank(val), 0)
  else:
    return np.isscalar(val)
def _has_docstring(f):
  return (f and hasattr(f, '__doc__') and isinstance(f.__doc__, str) and
          f.__doc__)
def _add_blank_line(s):
  if s.endswith('\n'):
    return s + '\n'
  else:
    return s + '\n\n'
def _np_signature(f):
  if not hasattr(inspect, 'signature'):
    return None
  if f is None:
    return None
  if not isinstance(f, np.ufunc):
    try:
      return inspect.signature(f)
    except ValueError:
      return None
  def names_from_num(prefix, n):
    if n <= 0:
      return []
    elif n == 1:
      return [prefix]
    else:
      return [prefix + str(i + 1) for i in range(n)]
  input_names = names_from_num('x', f.nin)
  output_names = names_from_num('out', f.nout)
  keyword_only_params = [('where', True), ('casting', 'same_kind'),
                         ('order', 'K'), ('dtype', None), ('subok', True),
                         ('signature', None), ('extobj', None)]
  params = []
  params += [
      inspect.Parameter(name, inspect.Parameter.POSITIONAL_ONLY)
      for name in input_names
  ]
  if f.nout > 1:
    params += [
        inspect.Parameter(
            name, inspect.Parameter.POSITIONAL_ONLY, default=None)
        for name in output_names
    ]
  params += [
      inspect.Parameter(
          'out',
          inspect.Parameter.POSITIONAL_OR_KEYWORD,
          default=None if f.nout == 1 else (None,) * f.nout)
  ]
  params += [
      inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=default)
      for name, default in keyword_only_params
  ]
  return inspect.Signature(params)
def _is_compatible_param_kind(a, b):
  def relax(k):
    if k in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.KEYWORD_ONLY):
      return inspect.Parameter.POSITIONAL_OR_KEYWORD
    return k
  return relax(a) == relax(b)
def _prepare_np_fun_name_and_fun(np_fun_name, np_fun):
  if np_fun_name is not None:
    assert isinstance(np_fun_name, str)
  if np_fun is not None:
    assert not isinstance(np_fun, str)
  if np_fun is None:
    assert np_fun_name is not None
    try:
      np_fun = getattr(np, str(np_fun_name))
    except AttributeError:
      np_fun = None
  if np_fun_name is None:
    assert np_fun is not None
    np_fun_name = np_fun.__name__
  return np_fun_name, np_fun
def _np_doc_helper(f, np_f, np_fun_name=None, unsupported_params=None,
                   link=None):
  assert np_f or np_fun_name
  if not np_fun_name:
    np_fun_name = np_f.__name__
  doc = 'TensorFlow variant of NumPy\'s `%s`.\n\n' % np_fun_name
  if unsupported_params:
    doc += 'Unsupported arguments: ' + ', '.join(
        '`' + name + '`' for name in unsupported_params) + '.\n\n'
  if _has_docstring(f):
    doc += f.__doc__
    doc = _add_blank_line(doc)
  doc = _add_np_doc(doc, np_fun_name, np_f, link=link)
  return doc
_np_doc_form = os.getenv('TF_NP_DOC_FORM', '1.16')
def get_np_doc_form():
  return _np_doc_form
def set_np_doc_form(value):
  r"""Selects the form of the original numpy docstrings.
  This function sets a global variable that controls how a tf-numpy symbol's
  docstring should refer to the original numpy docstring. If `value` is
  `'inlined'`, the numpy docstring will be verbatim copied into the tf-numpy
  docstring. Otherwise, a link to the original numpy docstring will be
  added. Which numpy version the link points to depends on `value`:
  * `'stable'`: the current stable version;
  * `'dev'`: the current development version;
  * pattern `\d+(\.\d+(\.\d+)?)?`: `value` will be treated as a version number,
    e.g. '1.16'.
  Args:
    value: the value to set the global variable to.
  """
  global _np_doc_form
  _np_doc_form = value
class Link:
  def __init__(self, v):
    self.value = v
class AliasOf:
  def __init__(self, v):
    self.value = v
class NoLink:
  pass
def generate_link(flag, np_fun_name):
  if flag == 'dev':
    template = 'https://numpy.org/devdocs/reference/generated/numpy.%s.html'
  elif flag == 'stable':
    template = (
        'https://numpy.org/doc/stable/reference/generated/numpy.%s.html')
  elif re.match(r'\d+(\.\d+(\.\d+)?)?$', flag):
    template = ('https://numpy.org/doc/' + flag +
                '/reference/generated/numpy.%s.html')
  else:
    return None
  return template % np_fun_name
_is_check_link = (os.getenv('TF_NP_CHECK_LINK', 'False') in
                  ('True', 'true', '1'))
def is_check_link():
  return _is_check_link
def set_check_link(value):
  global _is_check_link
  _is_check_link = value
def _add_np_doc(doc, np_fun_name, np_f, link):
  """Appends the numpy docstring to `doc`, according to `set_np_doc_form`.
  See `set_np_doc_form` for how it controls the form of the numpy docstring.
  Args:
    doc: the docstring to be appended to.
    np_fun_name: the name of the numpy function.
    np_f: (optional) the numpy function.
    link: (optional) which link to use. See `np_doc` for details.
  Returns:
    `doc` with numpy docstring appended.
  """
  flag = get_np_doc_form()
  if flag == 'inlined':
    if _has_docstring(np_f):
      doc += 'Documentation for `numpy.%s`:\n\n' % np_fun_name
      doc += np_f.__doc__.replace('>>>', '>')
  elif isinstance(flag, str):
    if link is None:
      url = generate_link(flag, np_fun_name)
    elif isinstance(link, AliasOf):
      url = generate_link(flag, link.value)
    elif isinstance(link, Link):
      url = link.value
    else:
      url = None
    if url is not None:
      if is_check_link():
        r = requests.head(url)
        if r.status_code != 200:
          raise ValueError(
              f'Check link failed at [{url}] with status code {r.status_code}. '
              f'Argument `np_fun_name` is {np_fun_name}.')
      doc += 'See the NumPy documentation for [`numpy.%s`](%s).' % (
          np_fun_name, url)
  return doc
_is_sig_mismatch_an_error = (
    os.getenv('TF_NP_SIG_MISMATCH_IS_ERROR', 'False') in ('True', 'true', '1'))
def is_sig_mismatch_an_error():
  return _is_sig_mismatch_an_error
def set_is_sig_mismatch_an_error(value):
  global _is_sig_mismatch_an_error
  _is_sig_mismatch_an_error = value
def np_doc(np_fun_name, np_fun=None, export=True, unsupported_params=None,
           link=None):
  """Attachs numpy docstring to a function.
  Args:
    np_fun_name: name for the np_fun symbol. At least one of np_fun or
      np_fun_name shoud be set.
    np_fun: (optional) the numpy function whose docstring will be used.
    export: whether to export this symbol under module
      `tf.experimental.numpy`. Note that if `export` is `True`, `np_fun` must be
      a function directly under the `numpy` module, not under any submodule of
      `numpy` (e.g. `numpy.random`).
    unsupported_params: (optional) the list of parameters not supported
      by tf.numpy.
    link: (optional) which link to use. If `None`, a default link generated from
      `np_fun_name` will be used. If an instance of `AliasOf`, `link.value` will
      be used in place of `np_fun_name` for the link generation. If an instance
      of `Link`, `link.value` will be used as the whole link. If an instance of
      `NoLink`, no link will be added.
  Returns:
    A function decorator that attaches the docstring from `np_fun` to the
    decorated function.
  """
  np_fun_name_orig, np_fun_orig = np_fun_name, np_fun
  np_fun_name, np_fun = _prepare_np_fun_name_and_fun(np_fun_name, np_fun)
  np_sig = _np_signature(np_fun)
  if unsupported_params is None:
    unsupported_params = []
  def decorator(f):
    if hasattr(inspect, 'signature') and np_sig is not None:
      try:
        sig = inspect.signature(f)
      except ValueError:
        sig = None
      if sig is not None:
        for name, param in sig.parameters.items():
          np_param = np_sig.parameters.get(name)
          if np_param is None:
            if is_sig_mismatch_an_error():
              raise TypeError(
                  f'Cannot find parameter {name} in the numpy function\'s '
                  f'signature (which has these parameters: '
                  f'{list(np_sig.parameters.keys())}). Argument `np_fun_name` '
                  f'is {np_fun_name_orig}. Argument `np_fun` is {np_fun_orig}.')
            else:
              continue
          if (is_sig_mismatch_an_error() and
              not _is_compatible_param_kind(param.kind, np_param.kind)):
            raise TypeError(
                f'Parameter {name} is of kind {param.kind} while in numpy it '
                f'is of kind {np_param.kind}. Argument `np_fun_name` is '
                f'{np_fun_name_orig}. Argument `np_fun` is {np_fun_orig}.')
          has_default = (param.default != inspect.Parameter.empty)
          np_has_default = (np_param.default != inspect.Parameter.empty)
          if is_sig_mismatch_an_error() and has_default != np_has_default:
            raise TypeError(
                'Parameter {} should{} have a default value. Argument '
                '`np_fun_name` is {}. Argument `np_fun` is {}.'.format(
                    name, '' if np_has_default else ' not', np_fun_name_orig,
                    np_fun_orig))
        for name in np_sig.parameters:
          if name not in sig.parameters:
            unsupported_params.append(name)
    f.__doc__ = _np_doc_helper(
        f, np_fun, np_fun_name=np_fun_name,
        unsupported_params=unsupported_params, link=link)
    if export:
      return np_export.np_export(np_fun_name)(f)
    else:
      return f
  return decorator
def np_doc_only(np_fun_name, np_fun=None, export=True):
  """Attachs numpy docstring to a function.
  This differs from np_doc in that it doesn't check for a match in signature.
  Args:
    np_fun_name: name for the np_fun symbol. At least one of np_fun or
      np_fun_name shoud be set.
    np_fun: (optional) the numpy function whose docstring will be used.
    export: whether to export this symbol under module
      `tf.experimental.numpy`. Note that if `export` is `True`, `np_f` must be a
      function directly under the `numpy` module, not under any submodule of
      `numpy` (e.g. `numpy.random`).
  Returns:
    A function decorator that attaches the docstring from `np_fun` to the
    decorated function.
  """
  np_fun_name, np_fun = _prepare_np_fun_name_and_fun(np_fun_name, np_fun)
  def decorator(f):
    f.__doc__ = _np_doc_helper(f, np_fun, np_fun_name=np_fun_name)
    if export:
      return np_export.np_export(np_fun_name)(f)
    else:
      return f
  return decorator
@np_doc('finfo')
def finfo(dtype):
  return np.finfo(_to_numpy_type(dtype))
def _maybe_get_dtype(x):
  if isinstance(x, numbers.Real):
    return x
  if isinstance(x, (core.Tensor, indexed_slices.IndexedSlices)):
    return _to_numpy_type(x.dtype)
  if isinstance(x, dtypes.DType):
    return x.as_numpy_dtype
  if isinstance(x, (list, tuple)):
    raise ValueError(
        f'Cannot find dtype for type inference from argument `x` of a sequence '
        f'type {type(x)}. For sequences, please call this function on each '
        f'element individually.')
  return x
@np_doc_only('result_type')
  arrays_and_dtypes = [
      _maybe_get_dtype(x) for x in nest.flatten(arrays_and_dtypes)
  ]
  if not arrays_and_dtypes:
    arrays_and_dtypes = [np.asarray([])]
  if dtype:
    return result_type(dtype)
  if isinstance(a, str):
    return np.unicode_
  elif isinstance(a, bytes):
    return np.bytes_
  return result_type(a)
  try:
  except ValueError:
    return result_type(t1, t2)
@np_doc('promote_types')
  type1 = _to_numpy_type(type1)
  type2 = _to_numpy_type(type2)
  return np_dtypes.canonicalize_dtype(np.promote_types(type1, type2))
def tf_broadcast(*args):
  if len(args) <= 1:
    return args
  sh = array_ops.shape(args[0])
  for arg in args[1:]:
    sh = array_ops.broadcast_dynamic_shape(sh, array_ops.shape(arg))
  return [array_ops.broadcast_to(arg, sh) for arg in args]
def get_static_value(x):
  if isinstance(x, core.Tensor) and (x.dtype.is_floating or x.dtype.is_complex):
    return None
  return tensor_util.constant_value(x)
def _maybe_static(x):
  value = get_static_value(x)
  if value is None:
    return x
  else:
    return value
def cond(pred, true_fn, false_fn):
  v = get_static_value(pred)
  if v is None:
    return control_flow_ops.cond(pred, true_fn, false_fn)
  if v:
    return true_fn()
  else:
    return false_fn()
def add(a, b):
  return _maybe_static(a) + _maybe_static(b)
def subtract(a, b):
  return _maybe_static(a) - _maybe_static(b)
def greater(a, b):
  return _maybe_static(a) > _maybe_static(b)
def greater_equal(a, b):
  return _maybe_static(a) >= _maybe_static(b)
def less_equal(a, b):
  return _maybe_static(a) <= _maybe_static(b)
def logical_and(a, b):
  a_value = get_static_value(a)
  if a_value is not None:
    if np.isscalar(a_value):
      if a_value:
        return _maybe_static(b)
      else:
        return a_value
    else:
      return a_value & _maybe_static(b)
  else:
    return a & _maybe_static(b)
def logical_or(a, b):
  a_value = get_static_value(a)
  if a_value is not None:
    if np.isscalar(a_value):
      if a_value:
        return a_value
      else:
        return _maybe_static(b)
    else:
      return a_value | _maybe_static(b)
  else:
    return a | _maybe_static(b)
def getitem(a, slice_spec):
  return _maybe_static(a)[slice_spec]
def reduce_all(input_tensor, axis=None, keepdims=False):
  v = get_static_value(input_tensor)
  if v is None:
    return math_ops.reduce_all(input_tensor, axis=axis, keepdims=keepdims)
  else:
    return v.all(axis=axis, keepdims=keepdims)
def reduce_any(input_tensor, axis=None, keepdims=False):
  v = get_static_value(input_tensor)
  if v is None:
    return math_ops.reduce_any(input_tensor, axis=axis, keepdims=keepdims)
  else:
    return v.any(axis=axis, keepdims=keepdims)
def tf_rank(t):
  r = t.shape.rank
  if r is not None:
    return r
  return array_ops.rank(t)
