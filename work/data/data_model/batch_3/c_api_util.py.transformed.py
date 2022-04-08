
import contextlib
from tensorflow.core.framework import api_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib
class ScopedTFStatus(object):
  __slots__ = ["status"]
  def __init__(self):
    self.status = c_api.TF_NewStatus()
  def __del__(self):
    if c_api is not None and c_api.TF_DeleteStatus is not None:
      c_api.TF_DeleteStatus(self.status)
class ScopedTFGraph(object):
  __slots__ = ["graph", "deleter"]
  def __init__(self):
    self.graph = c_api.TF_NewGraph()
    self.deleter = c_api.TF_DeleteGraph
  def __del__(self):
    self.deleter(self.graph)
class ScopedTFImportGraphDefOptions(object):
  __slots__ = ["options"]
  def __init__(self):
    self.options = c_api.TF_NewImportGraphDefOptions()
  def __del__(self):
    if c_api is not None and c_api.TF_DeleteImportGraphDefOptions is not None:
      c_api.TF_DeleteImportGraphDefOptions(self.options)
class ScopedTFImportGraphDefResults(object):
  __slots__ = ["results"]
  def __init__(self, results):
    self.results = results
  def __del__(self):
    if c_api is not None and c_api.TF_DeleteImportGraphDefResults is not None:
      c_api.TF_DeleteImportGraphDefResults(self.results)
class FunctionAlreadyGarbageCollectedError(Exception):
  def __init__(self, function_name):
    super(FunctionAlreadyGarbageCollectedError, self).__init__(
        "{} has already been garbage collected and cannot be called.".format(
            function_name))
class ScopedTFFunction(object):
  __slots__ = ["_func", "name", "deleter"]
  def __init__(self, func, name):
    self._func = func
    self.name = name
    self.deleter = c_api.TF_DeleteFunction
  @contextlib.contextmanager
  def get(self):
    if not self._func:
      raise FunctionAlreadyGarbageCollectedError(self.name)
    yield self._func
  def __del__(self):
    func = self._func
    if func:
      self._func = None
      self.deleter(func)
class ScopedTFBuffer(object):
  __slots__ = ["buffer"]
  def __init__(self, buf_string):
    self.buffer = c_api.TF_NewBufferFromString(compat.as_bytes(buf_string))
  def __del__(self):
    c_api.TF_DeleteBuffer(self.buffer)
class ApiDefMap(object):
  __slots__ = ["_api_def_map", "_op_per_name"]
  def __init__(self):
    op_def_proto = op_def_pb2.OpList()
    buf = c_api.TF_GetAllOpList()
    try:
      op_def_proto.ParseFromString(c_api.TF_GetBuffer(buf))
      self._api_def_map = c_api.TF_NewApiDefMap(buf)
    finally:
      c_api.TF_DeleteBuffer(buf)
    self._op_per_name = {}
    for op in op_def_proto.op:
      self._op_per_name[op.name] = op
  def __del__(self):
    if c_api is not None and c_api.TF_DeleteApiDefMap is not None:
      c_api.TF_DeleteApiDefMap(self._api_def_map)
  def put_api_def(self, text):
    c_api.TF_ApiDefMapPut(self._api_def_map, text, len(text))
  def get_api_def(self, op_name):
    api_def_proto = api_def_pb2.ApiDef()
    buf = c_api.TF_ApiDefMapGet(self._api_def_map, op_name, len(op_name))
    try:
      api_def_proto.ParseFromString(c_api.TF_GetBuffer(buf))
    finally:
      c_api.TF_DeleteBuffer(buf)
    return api_def_proto
  def get_op_def(self, op_name):
    if op_name in self._op_per_name:
      return self._op_per_name[op_name]
    raise ValueError(f"No op_def found for op name {op_name}.")
  def op_names(self):
    return self._op_per_name.keys()
@tf_contextlib.contextmanager
def tf_buffer(data=None):
  """Context manager that creates and deletes TF_Buffer.
  Example usage:
    with tf_buffer() as buf:
      ...
      proto_data = c_api.TF_GetBuffer(buf)
      graph_def.ParseFromString(compat.as_bytes(proto_data))
    with tf_buffer(some_string) as buf:
      c_api.TF_SomeFunction(buf)
  Args:
    data: An optional `bytes`, `str`, or `unicode` object. If not None, the
      yielded buffer will contain this data.
  Yields:
    Created TF_Buffer
  """
  if data:
    buf = c_api.TF_NewBufferFromString(compat.as_bytes(data))
  else:
    buf = c_api.TF_NewBuffer()
  try:
    yield buf
  finally:
    c_api.TF_DeleteBuffer(buf)
def tf_output(c_op, index):
  ret = c_api.TF_Output()
  ret.oper = c_op
  ret.index = index
  return ret
def tf_operations(graph):
  pos = 0
  c_op, pos = c_api.TF_GraphNextOperation(graph._c_graph, pos)
  while c_op is not None:
    yield c_op
    c_op, pos = c_api.TF_GraphNextOperation(graph._c_graph, pos)
def new_tf_operations(graph):
  for c_op in tf_operations(graph):
    try:
    except KeyError:
      yield c_op
