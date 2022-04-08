
from abc import abstractmethod
from contextlib import closing
import functools
import hashlib
import multiprocessing
import multiprocessing.dummy
import os
import queue
import random
import shutil
import tarfile
import threading
import time
import typing
import urllib
import weakref
import zipfile
import numpy as np
from tensorflow.python.framework import ops
from six.moves.urllib.request import urlopen
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.keras.utils.io_utils import path_to_string
from tensorflow.python.util.tf_export import keras_export
if sys.version_info[0] == 2:
  def urlretrieve(url, filename, reporthook=None, data=None):
    def chunk_read(response, chunk_size=8192, reporthook=None):
      content_type = response.info().get('Content-Length')
      total_size = -1
      if content_type is not None:
        total_size = int(content_type.strip())
      count = 0
      while True:
        chunk = response.read(chunk_size)
        count += 1
        if reporthook is not None:
          reporthook(count, chunk_size, total_size)
        if chunk:
          yield chunk
        else:
          break
    response = urlopen(url, data)
    with open(filename, 'wb') as fd:
      for chunk in chunk_read(response, reporthook=reporthook):
        fd.write(chunk)
else:
def is_generator_or_sequence(x):
  builtin_iterators = (str, list, tuple, dict, set, frozenset)
  if isinstance(x, (ops.Tensor, np.ndarray) + builtin_iterators):
    return False
  return (tf_inspect.isgenerator(x) or
          isinstance(x, Sequence) or
          isinstance(x, typing.Iterator))
def _extract_archive(file_path, path='.', archive_format='auto'):
  if archive_format is None:
    return False
  if archive_format == 'auto':
    archive_format = ['tar', 'zip']
  if isinstance(archive_format, str):
    archive_format = [archive_format]
  file_path = path_to_string(file_path)
  path = path_to_string(path)
  for archive_type in archive_format:
    if archive_type == 'tar':
      open_fn = tarfile.open
      is_match_fn = tarfile.is_tarfile
    if archive_type == 'zip':
      open_fn = zipfile.ZipFile
      is_match_fn = zipfile.is_zipfile
    if is_match_fn(file_path):
      with open_fn(file_path) as archive:
        try:
          archive.extractall(path)
        except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
          if os.path.exists(path):
            if os.path.isfile(path):
              os.remove(path)
            else:
              shutil.rmtree(path)
          raise
      return True
  return False
@keras_export('keras.utils.get_file')
def get_file(fname,
             origin,
             untar=False,
             md5_hash=None,
             file_hash=None,
             cache_subdir='datasets',
             hash_algorithm='auto',
             extract=False,
             archive_format='auto',
             cache_dir=None):
  """Downloads a file from a URL if it not already in the cache.
  By default the file at the url `origin` is downloaded to the
  cache_dir `~/.keras`, placed in the cache_subdir `datasets`,
  and given the filename `fname`. The final location of a file
  `example.txt` would therefore be `~/.keras/datasets/example.txt`.
  Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
  Passing a hash will verify the file after download. The command line
  programs `shasum` and `sha256sum` can compute the hash.
  Example:
  ```python
  path_to_downloaded_file = tf.keras.utils.get_file(
      "flower_photos",
      "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
      untar=True)
  ```
  Args:
      fname: Name of the file. If an absolute path `/path/to/file.txt` is
          specified the file will be saved at that location.
      origin: Original URL of the file.
      untar: Deprecated in favor of `extract` argument.
          boolean, whether the file should be decompressed
      md5_hash: Deprecated in favor of `file_hash` argument.
          md5 hash of the file for verification
      file_hash: The expected hash string of the file after download.
          The sha256 and md5 hash algorithms are both supported.
      cache_subdir: Subdirectory under the Keras cache dir where the file is
          saved. If an absolute path `/path/to/folder` is
          specified the file will be saved at that location.
      hash_algorithm: Select the hash algorithm to verify the file.
          options are `'md5'`, `'sha256'`, and `'auto'`.
          The default 'auto' detects the hash algorithm in use.
      extract: True tries extracting the file as an Archive, like tar or zip.
      archive_format: Archive format to try for extracting the file.
          Options are `'auto'`, `'tar'`, `'zip'`, and `None`.
          `'tar'` includes tar, tar.gz, and tar.bz files.
          The default `'auto'` corresponds to `['tar', 'zip']`.
          None or an empty list will return no matches found.
      cache_dir: Location to store cached files, when None it
          defaults to the default directory `~/.keras/`.
  Returns:
      Path to the downloaded file
  """
  if cache_dir is None:
    cache_dir = os.path.join(os.path.expanduser('~'), '.keras')
  if md5_hash is not None and file_hash is None:
    file_hash = md5_hash
    hash_algorithm = 'md5'
  datadir_base = os.path.expanduser(cache_dir)
  if not os.access(datadir_base, os.W_OK):
    datadir_base = os.path.join('/tmp', '.keras')
  datadir = os.path.join(datadir_base, cache_subdir)
  _makedirs_exist_ok(datadir)
  fname = path_to_string(fname)
  if untar:
    untar_fpath = os.path.join(datadir, fname)
    fpath = untar_fpath + '.tar.gz'
  else:
    fpath = os.path.join(datadir, fname)
  download = False
  if os.path.exists(fpath):
    if file_hash is not None:
      if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
        print('A local file was found, but it seems to be '
              'incomplete or outdated because the ' + hash_algorithm +
              ' file hash does not match the original value of ' + file_hash +
              ' so we will re-download the data.')
        download = True
  else:
    download = True
  if download:
    print('Downloading data from', origin)
    class ProgressTracker(object):
      progbar = None
    def dl_progress(count, block_size, total_size):
      if ProgressTracker.progbar is None:
        if total_size == -1:
          total_size = None
        ProgressTracker.progbar = Progbar(total_size)
      else:
        ProgressTracker.progbar.update(count * block_size)
    error_msg = 'URL fetch failure on {}: {} -- {}'
    try:
      try:
        urlretrieve(origin, fpath, dl_progress)
      except urllib.error.HTTPError as e:
        raise Exception(error_msg.format(origin, e.code, e.msg))
      except urllib.error.URLError as e:
        raise Exception(error_msg.format(origin, e.errno, e.reason))
    except (Exception, KeyboardInterrupt) as e:
      if os.path.exists(fpath):
        os.remove(fpath)
      raise
    ProgressTracker.progbar = None
  if untar:
    if not os.path.exists(untar_fpath):
      _extract_archive(fpath, datadir, archive_format='tar')
    return untar_fpath
  if extract:
    _extract_archive(fpath, datadir, archive_format)
  return fpath
def _makedirs_exist_ok(datadir):
def _resolve_hasher(algorithm, file_hash=None):
  if algorithm == 'sha256':
    return hashlib.sha256()
  if algorithm == 'auto' and file_hash is not None and len(file_hash) == 64:
    return hashlib.sha256()
  return hashlib.md5()
def _hash_file(fpath, algorithm='sha256', chunk_size=65535):
  """Calculates a file sha256 or md5 hash.
  Example:
  ```python
  _hash_file('/path/to/file.zip')
  'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
  ```
  Args:
      fpath: path to the file being validated
      algorithm: hash algorithm, one of `'auto'`, `'sha256'`, or `'md5'`.
          The default `'auto'` detects the hash algorithm in use.
      chunk_size: Bytes to read at a time, important for large files.
  Returns:
      The file hash
  """
  if isinstance(algorithm, str):
    hasher = _resolve_hasher(algorithm)
  else:
    hasher = algorithm
  with open(fpath, 'rb') as fpath_file:
    for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
      hasher.update(chunk)
  return hasher.hexdigest()
def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
  hasher = _resolve_hasher(algorithm, file_hash)
  if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
    return True
  else:
    return False
class ThreadsafeIter(object):
  def __init__(self, it):
    self.it = it
    self.lock = threading.Lock()
    self._exception = None
  def __iter__(self):
    return self
  def next(self):
    return self.__next__()
  def __next__(self):
    with self.lock:
      if self._exception:
      try:
        return next(self.it)
      except Exception as e:
        self._exception = e
        raise
def threadsafe_generator(f):
  @functools.wraps(f)
  def g(*a, **kw):
    return ThreadsafeIter(f(*a, **kw))
  return g
@keras_export('keras.utils.Sequence')
class Sequence(object):
  """Base object for fitting to a sequence of data, such as a dataset.
  Every `Sequence` must implement the `__getitem__` and the `__len__` methods.
  If you want to modify your dataset between epochs you may implement
  `on_epoch_end`.
  The method `__getitem__` should return a complete batch.
  Notes:
  `Sequence` are a safer way to do multiprocessing. This structure guarantees
  that the network will only train once
   on each sample per epoch which is not the case with generators.
  Examples:
  ```python
  from skimage.io import imread
  from skimage.transform import resize
  import numpy as np
  import math
  class CIFAR10Sequence(Sequence):
      def __init__(self, x_set, y_set, batch_size):
          self.x, self.y = x_set, y_set
          self.batch_size = batch_size
      def __len__(self):
          return math.ceil(len(self.x) / self.batch_size)
      def __getitem__(self, idx):
          batch_x = self.x[idx * self.batch_size:(idx + 1) *
          self.batch_size]
          batch_y = self.y[idx * self.batch_size:(idx + 1) *
          self.batch_size]
          return np.array([
              resize(imread(file_name), (200, 200))
                 for file_name in batch_x]), np.array(batch_y)
  ```
  """
  @abstractmethod
  def __getitem__(self, index):
    raise NotImplementedError
  @abstractmethod
  def __len__(self):
    raise NotImplementedError
  def on_epoch_end(self):
    pass
  def __iter__(self):
    for item in (self[i] for i in range(len(self))):
      yield item
def iter_sequence_infinite(seq):
  while True:
    for item in seq:
      yield item
_SHARED_SEQUENCES = {}
_SEQUENCE_COUNTER = None
_DATA_POOLS = weakref.WeakSet()
_WORKER_IDS = set()
_FORCE_THREADPOOL = False
_FORCE_THREADPOOL_LOCK = threading.RLock()
def dont_use_multiprocessing_pool(f):
  @functools.wraps(f)
  def wrapped(*args, **kwargs):
    with _FORCE_THREADPOOL_LOCK:
      global _FORCE_THREADPOOL
      old_force_threadpool, _FORCE_THREADPOOL = _FORCE_THREADPOOL, True
      out = f(*args, **kwargs)
      _FORCE_THREADPOOL = old_force_threadpool
      return out
  return wrapped
def get_pool_class(use_multiprocessing):
  global _FORCE_THREADPOOL
  if not use_multiprocessing or _FORCE_THREADPOOL:
  return multiprocessing.Pool
def get_worker_id_queue():
  global _WORKER_ID_QUEUE
  if _WORKER_ID_QUEUE is None:
    _WORKER_ID_QUEUE = multiprocessing.Queue()
  return _WORKER_ID_QUEUE
def init_pool(seqs):
  global _SHARED_SEQUENCES
  _SHARED_SEQUENCES = seqs
def get_index(uid, i):
  return _SHARED_SEQUENCES[uid][i]
@keras_export('keras.utils.SequenceEnqueuer')
class SequenceEnqueuer(object):
  """Base class to enqueue inputs.
  The task of an Enqueuer is to use parallelism to speed up preprocessing.
  This is done with processes or threads.
  Example:
  ```python
      enqueuer = SequenceEnqueuer(...)
      enqueuer.start()
      datas = enqueuer.get()
      for data in datas:
      enqueuer.stop()
  ```
  The `enqueuer.get()` should be an infinite stream of datas.
  """
  def __init__(self, sequence,
               use_multiprocessing=False):
    self.sequence = sequence
    self.use_multiprocessing = use_multiprocessing
    global _SEQUENCE_COUNTER
    if _SEQUENCE_COUNTER is None:
      try:
        _SEQUENCE_COUNTER = multiprocessing.Value('i', 0)
      except OSError:
        _SEQUENCE_COUNTER = 0
    if isinstance(_SEQUENCE_COUNTER, int):
      self.uid = _SEQUENCE_COUNTER
      _SEQUENCE_COUNTER += 1
    else:
      with _SEQUENCE_COUNTER.get_lock():
        self.uid = _SEQUENCE_COUNTER.value
        _SEQUENCE_COUNTER.value += 1
    self.workers = 0
    self.executor_fn = None
    self.queue = None
    self.run_thread = None
    self.stop_signal = None
  def is_running(self):
    return self.stop_signal is not None and not self.stop_signal.is_set()
  def start(self, workers=1, max_queue_size=10):
    """Starts the handler's workers.
    Args:
        workers: Number of workers.
        max_queue_size: queue size
            (when full, workers could block on `put()`)
    """
    if self.use_multiprocessing:
      self.executor_fn = self._get_executor_init(workers)
    else:
      self.executor_fn = lambda _: get_pool_class(False)(workers)
    self.workers = workers
    self.queue = queue.Queue(max_queue_size)
    self.stop_signal = threading.Event()
    self.run_thread = threading.Thread(target=self._run)
    self.run_thread.daemon = True
    self.run_thread.start()
  def _send_sequence(self):
    _SHARED_SEQUENCES[self.uid] = self.sequence
  def stop(self, timeout=None):
    """Stops running threads and wait for them to exit, if necessary.
    Should be called by the same thread which called `start()`.
    Args:
        timeout: maximum time to wait on `thread.join()`
    """
    self.stop_signal.set()
    with self.queue.mutex:
      self.queue.queue.clear()
      self.queue.unfinished_tasks = 0
      self.queue.not_full.notify()
    self.run_thread.join(timeout)
    _SHARED_SEQUENCES[self.uid] = None
  def __del__(self):
    if self.is_running():
      self.stop()
  @abstractmethod
  def _run(self):
    raise NotImplementedError
  @abstractmethod
  def _get_executor_init(self, workers):
    raise NotImplementedError
  @abstractmethod
  def get(self):
    """Creates a generator to extract data from the queue.
    Skip the data if it is `None`.
        Generator yielding tuples `(inputs, targets)`
            or `(inputs, targets, sample_weights)`.
    """
    raise NotImplementedError
@keras_export('keras.utils.OrderedEnqueuer')
class OrderedEnqueuer(SequenceEnqueuer):
  def __init__(self, sequence, use_multiprocessing=False, shuffle=False):
    super(OrderedEnqueuer, self).__init__(sequence, use_multiprocessing)
    self.shuffle = shuffle
  def _get_executor_init(self, workers):
    def pool_fn(seqs):
      pool = get_pool_class(True)(
          workers, initializer=init_pool_generator,
          initargs=(seqs, None, get_worker_id_queue()))
      _DATA_POOLS.add(pool)
      return pool
    return pool_fn
  def _wait_queue(self):
    while True:
      time.sleep(0.1)
      if self.queue.unfinished_tasks == 0 or self.stop_signal.is_set():
        return
  def _run(self):
    sequence = list(range(len(self.sequence)))
    while True:
      if self.shuffle:
        random.shuffle(sequence)
      with closing(self.executor_fn(_SHARED_SEQUENCES)) as executor:
        for i in sequence:
          if self.stop_signal.is_set():
            return
          self.queue.put(
              executor.apply_async(get_index, (self.uid, i)), block=True)
        self._wait_queue()
        if self.stop_signal.is_set():
          return
      self.sequence.on_epoch_end()
  def get(self):
    """Creates a generator to extract data from the queue.
    Skip the data if it is `None`.
    Yields:
        The next element in the queue, i.e. a tuple
        `(inputs, targets)` or
        `(inputs, targets, sample_weights)`.
    """
    while self.is_running():
      try:
        inputs = self.queue.get(block=True, timeout=5).get()
        if self.is_running():
          self.queue.task_done()
        if inputs is not None:
          yield inputs
      except queue.Empty:
        pass
        self.stop()
        raise e
def init_pool_generator(gens, random_seed=None, id_queue=None):
  global _SHARED_SEQUENCES
  _SHARED_SEQUENCES = gens
  worker_proc = multiprocessing.current_process()
  worker_proc.name = 'Keras_worker_{}'.format(worker_proc.name)
  if random_seed is not None:
    np.random.seed(random_seed + worker_proc.ident)
  if id_queue is not None:
    id_queue.put(worker_proc.ident, block=True, timeout=0.1)
def next_sample(uid):
  return next(_SHARED_SEQUENCES[uid])
@keras_export('keras.utils.GeneratorEnqueuer')
class GeneratorEnqueuer(SequenceEnqueuer):
  def __init__(self, generator,
               use_multiprocessing=False,
               random_seed=None):
    super(GeneratorEnqueuer, self).__init__(generator, use_multiprocessing)
    self.random_seed = random_seed
  def _get_executor_init(self, workers):
    def pool_fn(seqs):
      pool = get_pool_class(True)(
          workers, initializer=init_pool_generator,
          initargs=(seqs, self.random_seed, get_worker_id_queue()))
      _DATA_POOLS.add(pool)
      return pool
    return pool_fn
  def _run(self):
    with closing(self.executor_fn(_SHARED_SEQUENCES)) as executor:
      while True:
        if self.stop_signal.is_set():
          return
        self.queue.put(
            executor.apply_async(next_sample, (self.uid,)), block=True)
  def get(self):
    """Creates a generator to extract data from the queue.
    Skip the data if it is `None`.
    Yields:
        The next element in the queue, i.e. a tuple
        `(inputs, targets)` or
        `(inputs, targets, sample_weights)`.
    """
    try:
      while self.is_running():
        inputs = self.queue.get(block=True).get()
        self.queue.task_done()
        if inputs is not None:
          yield inputs
    except StopIteration:
      last_ones = []
      while self.queue.qsize() > 0:
        last_ones.append(self.queue.get(block=True))
      for f in last_ones:
        f.wait()
      last_ones = [future.get() for future in last_ones if future.successful()]
      for inputs in last_ones:
        if inputs is not None:
          yield inputs
      self.stop()
      if 'generator already executing' in str(e):
        raise RuntimeError(
            'Your generator is NOT thread-safe. '
            'Keras requires a thread-safe generator when '
            '`use_multiprocessing=False, workers > 1`. ')
      raise e
