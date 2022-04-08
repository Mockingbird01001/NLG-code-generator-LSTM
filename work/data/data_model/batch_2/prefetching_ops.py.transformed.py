
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import structure
from tensorflow.python.eager import function
from tensorflow.python.framework import device as framework_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util.tf_export import tf_export
@tf_export("data.experimental.prefetch_to_device")
def prefetch_to_device(device, buffer_size=None):
  """A transformation that prefetches dataset values to the given `device`.
  NOTE: Although the transformation creates a `tf.data.Dataset`, the
  transformation must be the final `Dataset` in the input pipeline.
  For example,
  >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
  >>> dataset = dataset.apply(tf.data.experimental.prefetch_to_device("/cpu:0"))
  >>> for element in dataset:
  ...   print(f'Tensor {element} is on device {element.device}')
  Tensor 1 is on device /job:localhost/replica:0/task:0/device:CPU:0
  Tensor 2 is on device /job:localhost/replica:0/task:0/device:CPU:0
  Tensor 3 is on device /job:localhost/replica:0/task:0/device:CPU:0
  Args:
    device: A string. The name of a device to which elements will be prefetched.
    buffer_size: (Optional.) The number of elements to buffer on `device`.
      Defaults to an automatically chosen value.
  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """
  def _apply_fn(dataset):
    return dataset.apply(
        copy_to_device(target_device=device)).prefetch(buffer_size)
  return _apply_fn
@tf_export("data.experimental.copy_to_device")
def copy_to_device(target_device, source_device="/cpu:0"):
  def _apply_fn(dataset):
    return _CopyToDeviceDataset(
        dataset, target_device=target_device, source_device=source_device)
  return _apply_fn
class _CopyToDeviceDataset(dataset_ops.UnaryUnchangedStructureDataset):
  def __init__(self, input_dataset, target_device, source_device="/cpu:0"):
    self._target_device = target_device
    spec = framework_device.DeviceSpec().from_string(self._target_device)
    self._is_gpu_target = (spec.device_type == "GPU")
    self._source_device_string = source_device
    self._source_device = ops.convert_to_tensor(source_device)
    wrap_ds_variant = gen_dataset_ops.wrap_dataset_variant(
    @function.defun()
    def _init_func():
      ds_variant = gen_dataset_ops.unwrap_dataset_variant(wrap_ds_variant)
      resource = gen_dataset_ops.anonymous_iterator(
      with ops.control_dependencies(
          [gen_dataset_ops.make_iterator(ds_variant, resource)]):
        return gen_dataset_ops.iterator_to_string_handle(resource)
    @function.defun()
    def _remote_init_func():
      return functional_ops.remote_call(
          target=self._source_device,
          args=init_func_concrete.captured_inputs,
          Tout=[dtypes.string],
          f=init_func_concrete)
    self._init_captured_args = self._init_func.captured_inputs
    @function.defun(input_signature=[tensor_spec.TensorSpec([], dtypes.string)])
    def _next_func(string_handle):
      with ops.device(self._source_device_string):
        iterator = iterator_ops.Iterator.from_string_handle(
            string_handle,
            dataset_ops.get_legacy_output_types(self),
            dataset_ops.get_legacy_output_shapes(self),
            dataset_ops.get_legacy_output_classes(self))
      return structure.to_tensor_list(self.element_spec, iterator.get_next())
    @function.defun_with_attributes(
        input_signature=[tensor_spec.TensorSpec([], dtypes.string)],
        attributes={"experimental_ints_on_device": True})
    def _remote_next_func(string_handle):
      return functional_ops.remote_call(
          target=self._source_device,
          args=[string_handle] + next_func_concrete.captured_inputs,
          f=next_func_concrete)
    self._next_captured_args = self._next_func.captured_inputs
    @function.defun(input_signature=[tensor_spec.TensorSpec([], dtypes.string)])
    def _finalize_func(string_handle):
      iterator_resource = gen_dataset_ops.iterator_from_string_handle_v2(
          string_handle,
      with ops.control_dependencies([
          resource_variable_ops.destroy_resource_op(
              iterator_resource, ignore_lookup_error=True)]):
        return array_ops.constant(0, dtypes.int64)
    @function.defun(input_signature=[tensor_spec.TensorSpec([], dtypes.string)])
    def _remote_finalize_func(string_handle):
      return functional_ops.remote_call(
          target=self._source_device,
          args=[string_handle] + finalize_func_concrete.captured_inputs,
          Tout=[dtypes.int64],
          f=finalize_func_concrete)
    )
    self._finalize_captured_args = self._finalize_func.captured_inputs
    g = ops.get_default_graph()
    self._init_func.add_to_graph(g)
    self._next_func.add_to_graph(g)
    self._finalize_func.add_to_graph(g)
    with ops.device(self._target_device):
      variant_tensor = gen_dataset_ops.generator_dataset(
          self._init_captured_args,
          self._next_captured_args,
          self._finalize_captured_args,
          init_func=self._init_func,
          next_func=self._next_func,
          finalize_func=self._finalize_func,
    super(_CopyToDeviceDataset, self).__init__(input_dataset, variant_tensor)
  def make_one_shot_iterator(self):
    if self._is_gpu_target:
      raise ValueError(
          "`make_one_shot_iterator` is not compatible with GPU execution. "
          "Please use `Dataset.make_initializable_iterator()` instead."
      )
    else:
      return super(_CopyToDeviceDataset, self).make_one_shot_iterator()
class _MapOnGpuDataset(dataset_ops.UnaryDataset):
  def __init__(self, input_dataset, map_func, use_inter_op_parallelism=True):
    self._input_dataset = input_dataset
    self._use_inter_op_parallelism = use_inter_op_parallelism
    self._map_func = structured_function.StructuredFunctionWrapper(
        map_func,
        self._transformation_name(),
        dataset=input_dataset,
        defun_kwargs={"experimental_ints_on_device": True})
    variant_tensor = ged_ops.experimental_map_dataset(
        self._map_func.function.captured_inputs,
        f=self._map_func.function,
        use_inter_op_parallelism=self._use_inter_op_parallelism,
        **self._flat_structure)
    super(_MapOnGpuDataset, self).__init__(input_dataset, variant_tensor)
  def _functions(self):
    return [self._map_func]
  @property
  def element_spec(self):
    return self._map_func.output_structure
  def _transformation_name(self):
    return "map_on_gpu()"
def map_on_gpu(map_func):
  """Maps `map_func` across the elements of this dataset.
  NOTE: This is a highly experimental version of `tf.data.Dataset.map` that runs
  `map_func` on GPU. It must be used after applying the
  `tf.data.experimental.copy_to_device` transformation with a GPU device
  argument.
  Args:
    map_func: A function mapping a nested structure of tensors (having shapes
      and types defined by `self.output_shapes` and `self.output_types`) to
      another nested structure of tensors.
  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """
  def _apply_fn(dataset):
    return _MapOnGpuDataset(dataset, map_func)
  return _apply_fn
