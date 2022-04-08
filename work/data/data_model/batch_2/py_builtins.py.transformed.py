
import functools
import inspect
import numpy as np
import six
from tensorflow.python.autograph.utils import py_func
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.util import lazy_loader
from tensorflow.python.util import nest
input_lib = lazy_loader.LazyLoader(
    'input_lib', globals(),
    'tensorflow.python.distribute.input_lib')
parallel_ops = lazy_loader.LazyLoader(
    'parallel_ops', globals(),
    'tensorflow.python.ops.parallel_for.control_flow_ops')
UNSPECIFIED = object()
def overload_of(f):
  if f in SUPPORTED_BUILTINS:
    return BUILTIN_FUNCTIONS_MAP[f.__name__]
  return f
def _find_originating_frame(caller_fn_scope, innermost=True):
  ctx_frame = inspect.currentframe()
  result = None
  while ctx_frame is not None:
    if ctx_frame.f_locals.get(caller_fn_scope.name, None) is caller_fn_scope:
      result = ctx_frame
      if innermost:
        break
    ctx_frame = ctx_frame.f_back
  assert result is not None, (
      'the conversion process should ensure the caller_fn_scope is always'
      ' found somewhere on the call stack')
  return result
def locals_in_original_context(caller_fn_scope):
  return _find_originating_frame(caller_fn_scope, innermost=True).f_locals
def globals_in_original_context(caller_fn_scope):
  return _find_originating_frame(caller_fn_scope, innermost=True).f_globals
def eval_in_original_context(f, args, caller_fn_scope):
  ctx_frame = _find_originating_frame(caller_fn_scope, innermost=True)
  args = (
      args[0],
      ctx_frame.f_globals if len(args) < 2 else args[1],
      ctx_frame.f_locals if len(args) < 3 else args[2],
  )
  return f(*args)
def super_in_original_context(f, args, caller_fn_scope):
  if args:
    return f(*args)
  ctx_frame = _find_originating_frame(caller_fn_scope, innermost=False)
  type_arg = ctx_frame.f_locals['__class__']
  self_arg_name = ctx_frame.f_code.co_varnames[0]
  self_arg = ctx_frame.f_locals[self_arg_name]
  return f(type_arg, self_arg)
def abs_(x):
  if tensor_util.is_tf_type(x):
    return _tf_abs(x)
  if isinstance(x, dataset_ops.DatasetV2):
    return _tf_dataset_abs(x)
  return _py_abs(x)
def _tf_abs(x):
  return math_ops.abs(x)
def _tf_dataset_abs(x):
  specs = nest.flatten(x.element_spec)
  if len(specs) == 1:
    return x.map(math_ops.abs, num_parallel_calls=dataset_ops.AUTOTUNE)
  return x.map(
      lambda *e: nest.map_structure(math_ops.abs, e),
      num_parallel_calls=dataset_ops.AUTOTUNE)
def _py_abs(x):
  return abs(x)
def float_(x=0):
  if tensor_util.is_tf_type(x):
    return _tf_float(x)
  return _py_float(x)
def _tf_float(x):
  if x.dtype == dtypes.string:
    return gen_parsing_ops.string_to_number(x, out_type=dtypes.float32)
  return math_ops.cast(x, dtype=dtypes.float32)
def _py_float(x):
  return float(x)
def int_(x=0, base=UNSPECIFIED):
  if tensor_util.is_tf_type(x):
    return _tf_int(x, base)
  return _py_int(x, base)
def _tf_int(x, base):
  if base not in (10, UNSPECIFIED):
    raise NotImplementedError('base {} not supported for int'.format(base))
  if x.dtype == dtypes.string:
    return gen_parsing_ops.string_to_number(x, out_type=dtypes.int32)
  return math_ops.cast(x, dtype=dtypes.int32)
def _py_int(x, base):
  if base is UNSPECIFIED:
    return int(x)
  return int(x, base)
def len_(s):
  if tensors.is_tensor_array(s):
    return _tf_tensor_array_len(s)
  elif tensors.is_tensor_list(s):
    return _tf_tensor_list_len(s)
  elif tensor_util.is_tf_type(s):
    return _tf_tensor_len(s)
  if isinstance(s, dataset_ops.DatasetV2):
    return _tf_dataset_len(s)
  return _py_len(s)
def _tf_tensor_array_len(s):
  return s.size()
def _tf_tensor_list_len(s):
  return list_ops.tensor_list_length(s)
def _tf_tensor_len(s):
  if s.shape.ndims and s.shape.dims[0].value is not None:
    return s.shape.dims[0].value
  shape = array_ops.shape(s)
  assert shape.shape, 'shape tensor of zero size? {}'.format(shape)
  if shape.shape[0] == 0:
    raise ValueError(
        'len requires a non-scalar tensor, got one of shape {}'.format(shape))
  if shape.shape.dims[0].value is not None:
    return array_ops.shape(s)[0]
  rank = array_ops.rank(s)
  def raise_zero_rank_error():
    msg = gen_string_ops.string_join(
        ['len requires non-zero rank, got ',
         gen_string_ops.as_string(rank)])
    with ops.control_dependencies([control_flow_ops.Assert(False, [msg])]):
      return constant_op.constant(0, dtype=dtypes.int32)
  return control_flow_ops.cond(rank > 0, lambda: array_ops.shape(s)[0],
                               raise_zero_rank_error)
def _tf_dataset_len(s):
  l = cardinality.cardinality(s)
  msg = gen_string_ops.string_join([
      'len requires dataset with definitive cardinality, got ',
      gen_string_ops.as_string(l)
  ])
  with ops.control_dependencies([
      control_flow_ops.Assert(
          math_ops.logical_and(
              math_ops.not_equal(l, cardinality.INFINITE),
              math_ops.not_equal(l, cardinality.UNKNOWN)), [msg])
  ]):
    l = array_ops.identity(l)
  return l
def _py_len(s):
  return len(s)
def print_(*objects, **kwargs):
  unknown_kwargs = tuple(
      set(kwargs.keys()) - set(('sep', 'end', 'file', 'flush')))
  if unknown_kwargs:
    raise ValueError('invalid keyword arguments: {}'.format(unknown_kwargs))
  if any(tensor_util.is_tf_type(o) for o in objects):
    return _tf_py_func_print(objects, kwargs)
  else:
    _py_print(*objects, **kwargs)
def _py_print(*objects, **kwargs):
  print(*objects, **kwargs)
def _tf_py_func_print(objects, kwargs):
  override_kwargs = {k: v for k, v in kwargs.items() if v is not UNSPECIFIED}
  if 'flush' not in override_kwargs:
    override_kwargs['flush'] = True
  def print_wrapper(*vals):
    vals = tuple(v.numpy() if tensor_util.is_tf_type(v) else v for v in vals)
    vals = tuple(v.decode('utf-8') if isinstance(v, bytes) else v for v in vals)
    six.print_(*vals, **override_kwargs)
  return py_func.wrap_py_func(
      print_wrapper, None, objects, use_dummy_return=True)
def range_(start_or_stop, stop=UNSPECIFIED, step=UNSPECIFIED):
  if any(tensor_util.is_tf_type(s) for s in (start_or_stop, stop, step)):
    return _tf_range(start_or_stop, stop, step)
  return _py_range(start_or_stop, stop, step)
def _tf_range(start_or_stop, stop, step):
  if step is not UNSPECIFIED:
    return math_ops.range(start_or_stop, stop, step)
  if stop is not UNSPECIFIED:
    stop = math_ops.maximum(start_or_stop, stop)
    return math_ops.range(start_or_stop, stop)
  start_or_stop = math_ops.maximum(start_or_stop, 0)
  return math_ops.range(start_or_stop)
def _py_range(start_or_stop, stop, step):
  if step is not UNSPECIFIED:
    return range(start_or_stop, stop, step)
  if stop is not UNSPECIFIED:
    return range(start_or_stop, stop)
  return range(start_or_stop)
def enumerate_(s, start=0):
  if isinstance(s, dataset_ops.DatasetV2):
    return _tf_dataset_enumerate(s, start)
  if isinstance(s,
                (input_lib.DistributedIterator, input_lib.DistributedDataset)):
    raise NotImplementedError(
        'use a for loop over the dataset and keep a separate counter')
  return _py_enumerate(s, start)
def _tf_dataset_enumerate(s, start=0):
  return s.enumerate(start)
def _py_enumerate(s, start=0):
  return enumerate(s, start)
def zip_(*iterables):
  if all(isinstance(x, dataset_ops.DatasetV2) for x in iterables):
    return _tf_dataset_zip(*iterables)
  return _py_zip(*iterables)
def _tf_dataset_zip(*iterables):
  return dataset_ops.DatasetV2.zip(iterables)
def _py_zip(*iterables):
  return zip(*iterables)
def map_(fn, *iterables):
  if all(isinstance(x, dataset_ops.DatasetV2) for x in iterables):
    return _tf_dataset_map(fn, *iterables)
  return _py_map(fn, *iterables)
def _tf_dataset_map(fn, *iterables):
  return dataset_ops.DatasetV2.zip(iterables).map(fn)
def _py_map(fn, *iterables):
  return map(fn, *iterables)
def next_(iterator, default=UNSPECIFIED):
  if isinstance(iterator, iterator_ops.OwnedIterator):
    return next_tf_iterator(iterator, default)
  return next_py(iterator, default)
def _verify_spec_compatible(input_name, spec_name, input_, spec):
  assert isinstance(spec, tensor_spec.TensorSpec)
  if input is None:
    raise ValueError('{} cannot be None'.format(input_name))
  if isinstance(input_, (bool, int, float, str, np.ndarray)):
    input_ = ops.convert_to_tensor_v2(input_)
  input_dtype = getattr(input_, 'dtype', None)
  if input_dtype != spec.dtype:
    input_dtype_str = 'no dtype' if input_dtype is None else str(input_dtype)
    raise TypeError(
        '{} must have the same dtype as {}. Expected {}, got {}'.format(
            input_name, spec_name, spec.dtype, input_dtype_str))
def _verify_structure_compatible(input_name, spec_name, input_, spec):
  try:
    nest.assert_same_structure(input_, spec, expand_composites=True)
  except (ValueError, TypeError) as e:
    raise TypeError(
        '{} must have the same element structure as {}.\n\n{}'.format(
            input_name, spec_name, str(e)))
  nest.map_structure(
      functools.partial(_verify_spec_compatible, input_name, spec_name), input_,
      spec)
def next_tf_iterator(iterator, default=UNSPECIFIED):
  if default is UNSPECIFIED:
    return next(iterator)
  opt_iterate = iterator.get_next_as_optional()
  _verify_structure_compatible('the default argument', 'the iterate', default,
                               iterator.element_spec)
  return control_flow_ops.cond(opt_iterate.has_value(), opt_iterate.get_value,
                               lambda: default)
def next_py(iterator, default=UNSPECIFIED):
  if default is UNSPECIFIED:
    return next(iterator)
  return next(iterator, default)
def filter_(function, iterable):
  if isinstance(iterable, dataset_ops.DatasetV2):
    return _tf_dataset_filter(function, iterable)
  return _py_filter(function, iterable)
def _tf_dataset_filter(function, iterable):
  return iterable.filter(function)
def _py_filter(function, iterable):
  return filter(function, iterable)
def any_(iterable):
  if isinstance(iterable, dataset_ops.DatasetV2):
    return _tf_dataset_any(iterable)
  return _py_any(iterable)
def _tf_dataset_any(iterable):
  specs = nest.flatten(iterable.element_spec)
  if len(specs) != 1 or specs[0].dtype != dtypes.bool:
    raise ValueError('in graph mode, the "any" builtin only supports datasets '
                     'that return bool scalars; got: {}'.format(
                         iterable.element_spec))
  ds = iterable.filter(lambda x: x)
  ds = ds.take(1)
  ds = ds.reduce(constant_op.constant(False, dtype=dtypes.bool), lambda _, y: y)
  return ds
def _py_any(iterable):
  return any(iterable)
def all_(iterable):
  if isinstance(iterable, dataset_ops.DatasetV2):
    return _tf_dataset_all(iterable)
  return _py_all(iterable)
def _tf_dataset_all(iterable):
  specs = nest.flatten(iterable.element_spec)
  if len(specs) != 1 or specs[0].dtype != dtypes.bool:
    raise ValueError('in graph mode, the "all" builtin only supports datasets '
                     'that return bool scalars; got: {}'.format(
                         iterable.element_spec))
  ds = iterable.filter(lambda x: math_ops.logical_not(x))
  ds = ds.take(1)
  ds = ds.reduce(constant_op.constant(True, dtype=dtypes.bool), lambda _, y: y)
  return ds
def _py_all(iterable):
  return all(iterable)
def sorted_(iterable, key=UNSPECIFIED, reverse=UNSPECIFIED):
  if tensor_util.is_tf_type(iterable):
    return _tf_sorted(iterable, key, reverse)
  return _py_sorted(iterable, key, reverse)
def _tf_sorted(iterable, key, reverse):
  if reverse is UNSPECIFIED:
    direction = 'ASCENDING'
  else:
    direction = 'DESCENDING'
  if key is not UNSPECIFIED:
    mapped = parallel_ops.vectorized_map(key, iterable)
    if mapped.shape.rank is not None and mapped.shape.rank != 1:
      raise ValueError('sort only supports only 1D tensors')
    with ops.control_dependencies([
        check_ops.assert_rank_v2(mapped, 1,
                                 'sort only supports only 1D tensors')
    ]):
      order = sort_ops.argsort(mapped, direction=direction)
      return array_ops.gather_v2(iterable, order)
  if iterable.shape.rank is not None and iterable.shape.rank != 1:
    raise ValueError('sort only supports only 1D tensors')
  with ops.control_dependencies([
      check_ops.assert_rank_v2(iterable, 1,
                               'sort only supports only 1D tensors')
  ]):
    return sort_ops.sort(iterable, direction=direction)
def _py_sorted(iterable, key, reverse):
  if key is not UNSPECIFIED and reverse is UNSPECIFIED:
    return sorted(iterable, key=key)
  if key is UNSPECIFIED and reverse is not UNSPECIFIED:
    return sorted(iterable, reverse=reverse)
  if key is not UNSPECIFIED and reverse is not UNSPECIFIED:
    return sorted(iterable, key=key, reverse=reverse)
  return sorted(iterable)
SUPPORTED_BUILTINS = (abs, float, int, len, print, range, enumerate, zip, map,
                      filter, any, all, sorted)
BUILTIN_FUNCTIONS_MAP = {
    'abs': abs_,
    'any': any_,
    'all': all_,
    'enumerate': enumerate_,
    'filter': filter_,
    'float': float_,
    'int': int_,
    'len': len_,
    'map': map_,
    'next': next_,
    'print': print_,
    'range': range_,
    'sorted': sorted_,
    'zip': zip_,
}
