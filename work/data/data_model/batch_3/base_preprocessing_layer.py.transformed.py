
import abc
import collections
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util.tf_export import keras_export
@keras_export('keras.layers.experimental.preprocessing.PreprocessingLayer')
class PreprocessingLayer(Layer, metaclass=abc.ABCMeta):
  """Base class for Preprocessing Layers.
  **Don't use this class directly: it's an abstract base class!** You may
  be looking for one of the many built-in
  [preprocessing layers](https://keras.io/guides/preprocessing_layers/)
  instead.
  Preprocessing layers are layers whose state gets computed before model
  training starts. They do not get updated during training.
  Most preprocessing layers implement an `adapt()` method for state computation.
  The `PreprocessingLayer` class is the base class you would subclass to
  implement your own preprocessing layers.
  Attributes:
    streaming: Whether a layer can be adapted multiple times without resetting
      the state of the layer.
  """
  _must_restore_from_config = True
  def __init__(self, streaming=True, **kwargs):
    super(PreprocessingLayer, self).__init__(**kwargs)
    self._streaming = streaming
    self._is_compiled = False
    self._is_adapted = False
    self._reset_state_impl = self.reset_state
    self.reset_state = self._reset_state_wrapper
    self._adapt_function = None
  @property
  def streaming(self):
    return self._streaming
  @property
  def is_adapted(self):
    return self._is_adapted
  def update_state(self, data):
    raise NotImplementedError
    raise NotImplementedError
  def merge_state(self, layers):
    raise NotImplementedError
  def finalize_state(self):
    pass
  def make_adapt_function(self):
    if self._adapt_function is not None:
      return self._adapt_function
    def adapt_step(iterator):
      data = next(iterator)
      self._adapt_maybe_build(data)
      self.update_state(data)
    if self._steps_per_execution.numpy().item() == 1:
      adapt_fn = adapt_step
    else:
      def adapt_fn(iterator):
        for _ in math_ops.range(self._steps_per_execution):
          adapt_step(iterator)
    if not self._run_eagerly:
      adapt_fn = def_function.function(adapt_fn)
    self._adapt_function = adapt_fn
    return self._adapt_function
  def compile(self, run_eagerly=None, steps_per_execution=None):
    if steps_per_execution is None:
      steps_per_execution = 1
    self._configure_steps_per_execution(steps_per_execution)
    if run_eagerly is None:
      run_eagerly = self.dynamic
    self._run_eagerly = run_eagerly
    self._is_compiled = True
  def adapt(self, data, batch_size=None, steps=None, reset_state=True):
    """Fits the state of the preprocessing layer to the data being passed.
    After calling `adapt` on a layer, a preprocessing layer's state will not
    update during training. In order to make preprocessing layers efficient in
    any distribution context, they are kept constant with respect to any
    compiled `tf.Graph`s that call the layer. This does not affect the layer use
    when adapting each layer only once, but if you adapt a layer multiple times
    you will need to take care to re-compile any compiled functions as follows:
     * If you are adding a preprocessing layer to a `keras.Model`, you need to
       call `model.compile` after each subsequent call to `adapt`.
     * If you are calling a preprocessing layer inside `tf.data.Dataset.map`,
       you should call `map` again on the input `tf.data.Dataset` after each
       `adapt`.
     * If you are using a `tf.function` directly which calls a preprocessing
       layer, you need to call `tf.function` again on your callable after
       each subsequent call to `adapt`.
    `tf.keras.Model` example with multiple adapts:
    >>> layer = tf.keras.layers.experimental.preprocessing.Normalization(
    ...     axis=None)
    >>> layer.adapt([0, 2])
    >>> model = tf.keras.Sequential(layer)
    >>> model.predict([0, 1, 2])
    array([-1.,  0.,  1.], dtype=float32)
    >>> layer.adapt([-1, 1])
    >>> model.predict([0, 1, 2])
    array([0., 1., 2.], dtype=float32)
    `tf.data.Dataset` example with multiple adapts:
    >>> layer = tf.keras.layers.experimental.preprocessing.Normalization(
    ...     axis=None)
    >>> layer.adapt([0, 2])
    >>> input_ds = tf.data.Dataset.range(3)
    >>> normalized_ds = input_ds.map(layer)
    >>> list(normalized_ds.as_numpy_iterator())
    [array([-1.], dtype=float32),
     array([0.], dtype=float32),
     array([1.], dtype=float32)]
    >>> layer.adapt([-1, 1])
    >>> list(normalized_ds.as_numpy_iterator())
    [array([0.], dtype=float32),
     array([1.], dtype=float32),
     array([2.], dtype=float32)]
    Arguments:
        data: The data to train on. It can be passed either as a tf.data
          Dataset, or as a numpy array.
        batch_size: Integer or `None`.
            Number of samples per state update.
            If unspecified, `batch_size` will default to 32.
            Do not specify the `batch_size` if your data is in the
            form of datasets, generators, or `keras.utils.Sequence` instances
            (since they generate batches).
        steps: Integer or `None`.
            Total number of steps (batches of samples)
            When training with input tensors such as
            TensorFlow data tensors, the default `None` is equal to
            the number of samples in your dataset divided by
            the batch size, or 1 if that cannot be determined. If x is a
            `tf.data` dataset, and 'steps' is None, the epoch will run until
            the input dataset is exhausted. When passing an infinitely
            repeating dataset, you must specify the `steps` argument. This
            argument is not supported with array inputs.
        reset_state: Optional argument specifying whether to clear the state of
          the layer at the start of the call to `adapt`, or whether to start
          from the existing state. This argument may not be relevant to all
          preprocessing layers: a subclass of PreprocessingLayer may choose to
          throw if 'reset_state' is set to False.
    """
    _disallow_inside_tf_function('adapt')
    if not version_utils.should_use_v2():
    if not self.streaming and self._is_adapted and not reset_state:
      raise ValueError('{} does not supporting calling `adapt` twice without '
                       'resetting the state.'.format(self.__class__.__name__))
    if not self._is_compiled:
    if self.built and reset_state:
      self.reset_state()
    data_handler = data_adapter.DataHandler(
        data,
        batch_size=batch_size,
        steps_per_epoch=steps,
        epochs=1,
        steps_per_execution=self._steps_per_execution,
        distribute=False)
    self._adapt_function = self.make_adapt_function()
    for _, iterator in data_handler.enumerate_epochs():
      with data_handler.catch_stop_iteration():
        for _ in data_handler.steps():
          self._adapt_function(iterator)
          if data_handler.should_sync:
            context.async_wait()
    self.finalize_state()
    self._is_adapted = True
  def _reset_state_wrapper(self):
    self._reset_state_impl()
    self._is_adapted = False
  @trackable.no_automatic_dependency_tracking
  def _configure_steps_per_execution(self, steps_per_execution):
    self._steps_per_execution = variables.Variable(
        steps_per_execution,
        dtype='int64',
        aggregation=variables.VariableAggregationV2.ONLY_FIRST_REPLICA)
  def _adapt_maybe_build(self, data):
    if not self.built:
      try:
        data_shape = data.shape
        data_shape_nones = tuple([None] * len(data.shape))
      except AttributeError:
        data_shape = None
        data_shape_nones = None
      batch_input_shape = getattr(self, '_batch_input_shape', None)
      if batch_input_shape is None:
        self._batch_input_shape = data_shape_nones
      self.build(data_shape)
      self.built = True
class CombinerPreprocessingLayer(PreprocessingLayer):
  def __init__(self, combiner, **kwargs):
    super(CombinerPreprocessingLayer, self).__init__(**kwargs)
    self.state_variables = collections.OrderedDict()
    self._combiner = combiner
    self._adapt_accumulator = None
    self._adapt_accumulator = None
  @trackable.no_automatic_dependency_tracking
  def update_state(self, data):
    if self._adapt_accumulator is None:
      self._adapt_accumulator = self._get_accumulator()
    self._adapt_accumulator = self._combiner.compute(data,
                                                     self._adapt_accumulator)
  def merge_state(self, layers):
    accumulators = ([self._get_accumulator()] +
    merged_accumulator = self._combiner.merge(accumulators)
    self._set_accumulator(merged_accumulator)
  def finalize_state(self):
    if self._adapt_accumulator is not None:
      self._set_accumulator(self._adapt_accumulator)
  def compile(self, run_eagerly=None, steps_per_execution=None):
    if run_eagerly is None:
      run_eagerly = True
    super(CombinerPreprocessingLayer, self).compile(
        run_eagerly=run_eagerly, steps_per_execution=steps_per_execution)
  def adapt(self, data, batch_size=None, steps=None, reset_state=True):
    if not reset_state:
      self._adapt_accumulator = self._combiner.restore(self._restore_updates())
    super(CombinerPreprocessingLayer, self).adapt(
        data, batch_size=batch_size, steps=steps, reset_state=reset_state)
  def _add_state_variable(self,
                          name,
                          shape,
                          dtype,
                          initializer=None,
                          partitioner=None,
                          use_resource=None,
                          **kwargs):
    """Add a variable that can hold state which is updated during adapt().
    Args:
      name: Variable name.
      shape: Variable shape. Defaults to scalar if unspecified.
      dtype: The type of the variable. Defaults to `self.dtype` or `float32`.
      initializer: initializer instance (callable).
      partitioner: Partitioner to be passed to the `Trackable` API.
      use_resource: Whether to use `ResourceVariable`
      **kwargs: Additional keyword arguments. Accepted values are `getter` and
        `collections`.
    Returns:
      The created variable.
    """
    weight = self.add_weight(
        name=name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        regularizer=None,
        trainable=False,
        constraint=None,
        partitioner=partitioner,
        use_resource=use_resource,
        **kwargs)
    self.state_variables[name] = weight
    return weight
  def _restore_updates(self):
    data_dict = {}
    for name, var in self.state_variables.items():
      data_dict[name] = var.numpy()
    return data_dict
  def _get_accumulator(self):
    if self._is_adapted:
      return self._combiner.restore(self._restore_updates())
    else:
      return None
  def _set_accumulator(self, accumulator):
    updates = self._combiner.extract(accumulator)
    self._set_state_variables(updates)
  def _set_state_variables(self, updates):
    """Directly update the internal state of this Layer.
    This method expects a string-keyed dict of {state_variable_name: state}. The
    precise nature of the state, and the names associated, are describe by
    the subclasses of CombinerPreprocessingLayer.
    Args:
      updates: A string keyed dict of weights to update.
    Raises:
      RuntimeError: if 'build()' was not called before 'set_processing_state'.
    """
    if not self.built:
      raise RuntimeError('_set_state_variables() must be called after build().')
    with ops.init_scope():
      for var_name, value in updates.items():
        self.state_variables[var_name].assign(value)
def convert_to_list(values, sparse_default_value=None):
  if tf_utils.is_ragged(values):
    if (isinstance(values, ragged_tensor.RaggedTensor) and
        not context.executing_eagerly()):
      values = backend.get_session(values).run(values)
    values = values.to_list()
  if isinstance(values,
                (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue)):
    if sparse_default_value is None:
      if dtypes.as_dtype(values.values.dtype) == dtypes.string:
        sparse_default_value = ''
      else:
        sparse_default_value = -1
    dense_tensor = sparse_ops.sparse_tensor_to_dense(
        values, default_value=sparse_default_value)
    values = backend.get_value(dense_tensor)
  if isinstance(values, ops.Tensor):
    values = backend.get_value(values)
  if isinstance(values, np.ndarray):
    values = values.tolist()
  return values
class Combiner(object):
  """Functional object that defines a shardable computation.
  This object defines functions required to create and manipulate data objects.
  These data objects, referred to below as 'accumulators', are computation-
  specific and may be implemented alongside concrete subclasses of Combiner
  (if necessary - some computations may be simple enough that standard Python
  types can be used as accumulators).
  The intent for this class is that by describing computations in this way, we
  can arbitrarily shard a dataset, perform computations on a subset, and then
  merge the computation into a final result. This enables distributed
  computation.
  The combiner itself does not own any state - all computational state is owned
  by the accumulator objects. This is so that we can have an arbitrary number of
  Combiners (thus sharding the computation N ways) without risking any change
  to the underlying computation. These accumulator objects are uniquely
  associated with each Combiner; a Combiner defines what the accumulator object
  should be and will only work with accumulators of that type.
  """
  __metaclass__ = abc.ABCMeta
  def __repr__(self):
    return '<{}>'.format(self.__class__.__name__)
  @abc.abstractmethod
  def compute(self, batch_values, accumulator=None):
    """Compute a step in this computation, returning a new accumulator.
    This method computes a step of the computation described by this Combiner.
    If an accumulator is passed, the data in that accumulator is also used; so
    compute(batch_values) results in f(batch_values), while
    compute(batch_values, accumulator) results in
    merge(f(batch_values), accumulator).
    Args:
      batch_values: A list of ndarrays representing the values of the inputs for
        this step of the computation.
      accumulator: the current accumulator. Can be None.
    Returns:
      An accumulator that includes the passed batch of inputs.
    """
    pass
  @abc.abstractmethod
  def merge(self, accumulators):
    """Merge several accumulators to a single accumulator.
    This method takes the partial values in several accumulators and combines
    them into a single accumulator. This computation must not be order-specific
    (that is, merge([a, b]) must return the same result as merge([b, a]).
    Args:
      accumulators: the accumulators to merge, as a list.
    Returns:
      A merged accumulator.
    """
    pass
  @abc.abstractmethod
  def extract(self, accumulator):
    pass
  @abc.abstractmethod
  def restore(self, output):
    """Create an accumulator based on 'output'.
    This method creates a new accumulator with identical internal state to the
    one used to create the data in 'output'. This means that if you do
    output_data = combiner.extract(accumulator_1)
    accumulator_2 = combiner.restore(output_data)
    then accumulator_1 and accumulator_2 will have identical internal state, and
    computations using either of them will be equivalent.
    Args:
      output: The data output from a previous computation. Should be in the same
        form as provided by 'extract_output'.
    Returns:
      A new accumulator.
    """
    pass
  @abc.abstractmethod
  def serialize(self, accumulator):
    pass
  @abc.abstractmethod
  def deserialize(self, encoded_accumulator):
    """Deserialize an accumulator received from 'serialize()'.
    This function deserializes an accumulator serialized by 'serialize()'.
    Args:
      encoded_accumulator: A byte string representing an accumulator.
    Returns:
      The accumulator represented by the passed byte_string.
    """
    pass
def _disallow_inside_tf_function(method_name):
  if ops.inside_function():
    error_msg = (
        'Detected a call to `PreprocessingLayer.{method_name}` inside a '
        '`tf.function`. `PreprocessingLayer.{method_name} is a high-level '
        'endpoint that manages its own `tf.function`. Please move the call '
        'to `PreprocessingLayer.{method_name}` outside of all enclosing '
        '`tf.function`s. Note that you can call a `PreprocessingLayer` '
        'directly on `Tensor`s inside a `tf.function` like: `layer(x)`, '
        'or update its state like: `layer.update_state(x)`.').format(
            method_name=method_name)
    raise RuntimeError(error_msg)
