
import collections
import inspect
import threading
from tensorflow.python.util import _tf_stack
_get_thread_key = threading.get_ident
_source_mapper_stacks = collections.defaultdict(lambda: [SentinelMapper()])
_source_filter_stacks = collections.defaultdict(lambda: [SentinelFilter()])
class StackTraceTransform(object):
  _thread_key = None
  def __enter__(self):
    if self._thread_key is None:
      self._thread_key = _get_thread_key()
    else:
      assert self._thread_key == _get_thread_key(), 'Shared across threads?'
    stack = self._stack_dict[self._thread_key]
    self.parent = stack[-1]
    stack.append(self)
    self.update()
    return self
  def __exit__(self, unused_type, unused_value, unused_traceback):
    top = self._stack_dict[self._thread_key].pop()
    assert top is self, 'Concurrent access?'
  def update(self):
    raise NotImplementedError('subclasses need to override this')
class StackTraceMapper(StackTraceTransform):
  _stack_dict = _source_mapper_stacks
  def __init__(self):
    self.internal_map = _tf_stack.PyBindSourceMap()
  def update(self):
    self.internal_map.update_to(tuple(self.get_effective_source_map().items()))
  def get_effective_source_map(self):
    raise NotImplementedError('subclasses need to override this')
EMPTY_DICT = {}
class SentinelMapper(StackTraceMapper):
  def get_effective_source_map(self):
    return EMPTY_DICT
class StackTraceFilter(StackTraceTransform):
  _stack_dict = _source_filter_stacks
  def __init__(self):
    self.internal_set = _tf_stack.PyBindFileSet()
  def update(self):
    self.internal_set.update_to(set(self.get_filtered_filenames()))
  def get_filtered_filenames(self):
    raise NotImplementedError('subclasses need to override this')
EMPTY_SET = frozenset()
class SentinelFilter(StackTraceFilter):
  def get_filtered_filenames(self):
    return EMPTY_SET
class CurrentModuleFilter(StackTraceFilter):
  def __init__(self):
    super().__init__()
    filter_filename = None
    outer_f = None
    f = inspect.currentframe()
    try:
      if f is not None:
        outer_f = f.f_back
        if outer_f is not None:
          filter_filename = inspect.getsourcefile(outer_f)
      self._filename = filter_filename
      self._cached_set = None
    finally:
      del f
      del outer_f
  def get_filtered_filenames(self):
    if self._cached_set is not None:
      return self._cached_set
    filtered_filenames = frozenset((self._filename,))
    if self.parent is not None:
      filtered_filenames |= self.parent.get_filtered_filenames()
    self._cached_set = filtered_filenames
    return filtered_filenames
def extract_stack():
  thread_key = _get_thread_key()
  return _tf_stack.extract_stack(
      _source_mapper_stacks[thread_key][-1].internal_map,
      _source_filter_stacks[thread_key][-1].internal_set)
def extract_stack_for_node(node):
  thread_key = _get_thread_key()
  return _tf_stack.extract_stack_for_node(
      _source_mapper_stacks[thread_key][-1].internal_map,
      _source_filter_stacks[thread_key][-1].internal_set, node)
StackSummary = _tf_stack.StackTraceWrapper
FrameSummary = _tf_stack.StackFrame
