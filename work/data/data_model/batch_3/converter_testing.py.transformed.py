
import contextlib
import imp
import inspect
import sys
import six
from tensorflow.python.autograph.core import config
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.impl import api
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
def allowlist(f):
  if 'allowlisted_module_for_testing' not in sys.modules:
    allowlisted_mod = imp.new_module('allowlisted_module_for_testing')
    sys.modules['allowlisted_module_for_testing'] = allowlisted_mod
    config.CONVERSION_RULES = (
        (config.DoNotConvert('allowlisted_module_for_testing'),) +
        config.CONVERSION_RULES)
  f.__module__ = 'allowlisted_module_for_testing'
def is_inside_generated_code():
  frame = inspect.currentframe()
  try:
    frame = frame.f_back
    internal_stack_functions = ('converted_call', '_call_unconverted')
    while (frame is not None and
           frame.f_code.co_name in internal_stack_functions):
      frame = frame.f_back
    if frame is None:
      return False
    return 'ag__' in frame.f_locals
  finally:
    del frame
class TestingTranspiler(api.PyToTF):
  def __init__(self, converters, ag_overrides):
    super(TestingTranspiler, self).__init__()
    if isinstance(converters, (list, tuple)):
      self._converters = converters
    else:
      self._converters = (converters,)
    self.transformed_ast = None
    self._ag_overrides = ag_overrides
  def get_extra_locals(self):
    retval = super(TestingTranspiler, self).get_extra_locals()
    if self._ag_overrides:
      modified_ag = imp.new_module('fake_autograph')
      modified_ag.__dict__.update(retval['ag__'].__dict__)
      modified_ag.__dict__.update(self._ag_overrides)
      retval['ag__'] = modified_ag
    return retval
  def transform_ast(self, node, ctx):
    node = self.initial_analysis(node, ctx)
    for c in self._converters:
      node = c.transform(node, ctx)
    self.transformed_ast = node
    self.transform_ctx = ctx
    return node
class TestCase(test.TestCase):
  def setUp(self):
    self.graph = ops.Graph().as_default()
    self.graph.__enter__()
  def tearDown(self):
    self.graph.__exit__(None, None, None)
  @contextlib.contextmanager
  def assertPrints(self, expected_result):
    try:
      out_capturer = six.StringIO()
      sys.stdout = out_capturer
      yield
      self.assertEqual(out_capturer.getvalue(), expected_result)
    finally:
      sys.stdout = sys.__stdout__
  def transform(
      self, f, converter_module, include_ast=False, ag_overrides=None):
    program_ctx = converter.ProgramContext(
        options=converter.ConversionOptions(recursive=True),
        autograph_module=api)
    tr = TestingTranspiler(converter_module, ag_overrides)
    transformed, _, _ = tr.transform_function(f, program_ctx)
    if include_ast:
      return transformed, tr.transformed_ast, tr.transform_ctx
    return transformed
