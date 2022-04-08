
from tensorflow.python.framework import test_util
from tensorflow.python.framework import traceable_stack
from tensorflow.python.platform import googletest
from tensorflow.python.util import tf_inspect as inspect
_LOCAL_OBJECT = lambda x: x
_THIS_FILENAME = inspect.getsourcefile(_LOCAL_OBJECT)
class TraceableObjectTest(test_util.TensorFlowTestCase):
  def testSetFilenameAndLineFromCallerUsesCallersStack(self):
    t_obj = traceable_stack.TraceableObject(17)
    placeholder = lambda x: x
    result = t_obj.set_filename_and_line_from_caller()
    expected_lineno = inspect.getsourcelines(placeholder)[1] + 1
    self.assertEqual(expected_lineno, t_obj.lineno)
    self.assertEqual(_THIS_FILENAME, t_obj.filename)
    self.assertEqual(t_obj.SUCCESS, result)
  def testSetFilenameAndLineFromCallerRespectsOffset(self):
    def call_set_filename_and_line_from_caller(t_obj):
      return t_obj.set_filename_and_line_from_caller(offset=1)
    t_obj = traceable_stack.TraceableObject(None)
    placeholder = lambda x: x
    result = call_set_filename_and_line_from_caller(t_obj)
    expected_lineno = inspect.getsourcelines(placeholder)[1] + 1
    self.assertEqual(expected_lineno, t_obj.lineno)
    self.assertEqual(t_obj.SUCCESS, result)
  def testSetFilenameAndLineFromCallerHandlesRidiculousOffset(self):
    t_obj = traceable_stack.TraceableObject('The quick brown fox.')
    result = t_obj.set_filename_and_line_from_caller(offset=300)
    self.assertEqual(t_obj.HEURISTIC_USED, result)
class TraceableStackTest(test_util.TensorFlowTestCase):
  def testPushPeekPopObj(self):
    t_stack = traceable_stack.TraceableStack()
    t_stack.push_obj(42.0)
    t_stack.push_obj('hope')
    expected_lifo_peek = ['hope', 42.0]
    self.assertEqual(expected_lifo_peek, list(t_stack.peek_objs()))
    self.assertEqual('hope', t_stack.pop_obj())
    self.assertEqual(42.0, t_stack.pop_obj())
  def testPushPeekTopObj(self):
    t_stack = traceable_stack.TraceableStack()
    t_stack.push_obj(42.0)
    t_stack.push_obj('hope')
    self.assertEqual('hope', t_stack.peek_top_obj())
  def testPushPopPreserveLifoOrdering(self):
    t_stack = traceable_stack.TraceableStack()
    t_stack.push_obj(0)
    t_stack.push_obj(1)
    t_stack.push_obj(2)
    t_stack.push_obj(3)
    obj_3 = t_stack.pop_obj()
    obj_2 = t_stack.pop_obj()
    obj_1 = t_stack.pop_obj()
    obj_0 = t_stack.pop_obj()
    self.assertEqual(3, obj_3)
    self.assertEqual(2, obj_2)
    self.assertEqual(1, obj_1)
    self.assertEqual(0, obj_0)
  def testPushObjSetsFilenameAndLineInfoForCaller(self):
    t_stack = traceable_stack.TraceableStack()
    placeholder_1 = lambda x: x
    t_stack.push_obj(1)
    def call_push_obj(obj):
      t_stack.push_obj(obj, offset=1)
    placeholder_2 = lambda x: x
    call_push_obj(2)
    expected_lineno_1 = inspect.getsourcelines(placeholder_1)[1] + 1
    expected_lineno_2 = inspect.getsourcelines(placeholder_2)[1] + 1
    t_obj_2, t_obj_1 = t_stack.peek_traceable_objs()
    self.assertEqual(expected_lineno_2, t_obj_2.lineno)
    self.assertEqual(expected_lineno_1, t_obj_1.lineno)
if __name__ == '__main__':
  googletest.main()
