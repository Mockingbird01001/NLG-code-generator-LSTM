
from typing import Callable, Optional, Text, Union
from tensorflow.python.data.experimental.ops import interleave_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import readers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import functional_ops
def _TextLineDataset(filename: Text) -> dataset_ops.Dataset:
  dataset = readers.TextLineDataset(filename, buffer_size=buffer_size)
  return dataset
def _TFRecordDataset(filename: Text) -> dataset_ops.Dataset:
  dataset = readers.TFRecordDataset(filename, buffer_size=buffer_size)
  return dataset
_FILETYPE_MAP = {
    'tfrecord': _TFRecordDataset,
    'textline': _TextLineDataset,
    'text': _TextLineDataset,
}
def StreamingFilesDataset(
    files: Union[Text, dataset_ops.Dataset],
    filetype: Optional[Union[Text, Callable[[Text],
                                            dataset_ops.Dataset]]] = None,
    file_reader_job: Optional[Text] = None,
    worker_job: Optional[Text] = None,
    num_epochs: Optional[int] = None,
    filename_shuffle_buffer_size: Optional[Union[int, bool]] = None,
    num_parallel_reads: Optional[int] = None,
    batch_transfer_size: Optional[Union[int, bool]] = None,
    sloppy: bool = True) -> dataset_ops.Dataset:
  """StreamingFilesDataset constructs a dataset to stream from workers (GCE VM).
  Because Cloud TPUs are allocated over the network, a Cloud TPU cannot read
  files local to your GCE VM. In order to train using files stored on your local
  VM (e.g. on local SSD for extreme performance), use the StreamingFilesDataset
  helper to generate a dataset to feed your Cloud TPU with files from your GCE
  VM.
  The resulting dataset may return an OutOfRangeError if there are no files
  found as a result of the fileglob expansion.
  Note: StreamingFilesDataset assumes that the session is using a
  TPUClusterResolver and has therefore a worker and a coordinator job. File
  loading will be done on the coordinator job.
  Args:
    files: A string glob to match files, or a `tf.data.Dataset` generating file
      names.
    filetype: A string (one of 'tfrecord', or 'textline') or a single-argument
      TensorFlow function that when given a filename returns a dataset.
    file_reader_job: An optional string that corresponds to the job that should
      perform the file reads.
    worker_job: An optional string that corresponds to the job that should
      process the tensors (i.e. your GPU or TPU worker).
    num_epochs: The number of epochs through the training set that should be
      generated. By default, it will repeat infinitely.
    filename_shuffle_buffer_size: An optional integer whose value controls the
      shuffling of the file names. If you would like to read from the files in
      the same order, set to 0 or False.
    num_parallel_reads: An optional integer controlling the number of files to
      read from concurrently. (Set to 1 for no parallelism.)
    batch_transfer_size: An optional integer controlling the batching used to
      amortize the remote function invocation overhead. Set to a very large
      number to increase throughput. Set to a very small number to reduce memory
      consumption. Set to False to skip batching.
    sloppy: (Optional.) If `False`, read input data while maintaining a
      deterministic order. (This may have significant performance impacts.)
      sloppy defaults to: True.
  Returns:
    A `tf.data.Dataset` with an infinite stream of elements generated by a
    parallel interleaving of the set of files matched (or generated) by `files`
    with a type is the output of the dataset specified by `filetype`.
  Raises:
    ValueError: if any argument is not of the expected type.
  """
  if filetype is None:
    filetype = 'tfrecord'
  if isinstance(filetype, str):
    if filetype not in _FILETYPE_MAP:
      raise ValueError(
          f'Unexpected filetype. Received: {filetype}. Expected one of '
          f'{list(_FILETYPE_MAP.keys())}')
    reader_fn = _FILETYPE_MAP[filetype]
  elif callable(filetype):
    reader_fn = filetype
  else:
    raise ValueError(f'Argument `filetype` should be a string or a callable. '
                     f'Received: {filetype} of type {type(filetype)}.')
  file_reader_job = file_reader_job or 'coordinator'
  worker_job = worker_job or 'worker'
  if filename_shuffle_buffer_size is None:
    filename_shuffle_buffer_size = 4096
  num_parallel_reads = num_parallel_reads or 8
  if batch_transfer_size is None:
    batch_transfer_size = 256
  if file_reader_job == 'coordinator':
    file_reader_device = '/job:coordinator/task:0'
  else:
    file_reader_device = '/job:%s' % file_reader_job
  with ops.device(file_reader_device):
    if isinstance(files, str):
      source_dataset = dataset_ops.Dataset.list_files(files)
    elif isinstance(files, dataset_ops.DatasetV2):
      source_dataset = files
    else:
      raise ValueError(
          'Argument `files` should be a string or a `tf.data.Dataset` '
          f'instance. Received: {files}')
    if filename_shuffle_buffer_size:
      source_dataset = source_dataset.shuffle(
          buffer_size=filename_shuffle_buffer_size)
    source_dataset = source_dataset.apply(
        interleave_ops.parallel_interleave(
            reader_fn, cycle_length=num_parallel_reads, sloppy=sloppy))
    source_dataset = source_dataset.repeat(num_epochs)
    if batch_transfer_size:
      source_dataset = source_dataset.batch(batch_transfer_size)
    source_dataset = source_dataset.prefetch(1)
    source_iterator = dataset_ops.make_one_shot_iterator(source_dataset)
    source_handle = source_iterator.string_handle()
  @function.Defun(dtypes.string)
  def LoadingFunc(h):
    remote_iterator = iterator_ops.Iterator.from_string_handle(
        h, dataset_ops.get_legacy_output_types(source_dataset),
        dataset_ops.get_legacy_output_shapes(source_dataset))
    return remote_iterator.get_next()
  def MapFn(unused_input):
    source_dataset_output_types = dataset_ops.get_legacy_output_types(
        source_dataset)
    if isinstance(source_dataset_output_types, dtypes.DType):
      output_types = [source_dataset_output_types]
    elif isinstance(source_dataset_output_types, (list, tuple)):
      output_types = source_dataset_output_types
    else:
      raise ValueError('Source dataset has invalid output types. Only '
                       'list/tuples or TensorFlow tensor types are accepted.')
    remote_calls = functional_ops.remote_call(
        args=[source_handle],
        Tout=output_types,
        f=LoadingFunc,
        target='/job:%s/replica:0/task:0/cpu:0' % file_reader_job)
    if len(remote_calls) == 1:
      return remote_calls[0]
    else:
      return remote_calls
  with ops.device('/job:%s' % worker_job):
    output_dataset = dataset_ops.Dataset.range(2).repeat().map(
        MapFn, num_parallel_calls=4 if sloppy else None)
    output_dataset = output_dataset.prefetch(1)
    if batch_transfer_size:
      output_dataset = output_dataset.unbatch().prefetch(1)
  return output_dataset
