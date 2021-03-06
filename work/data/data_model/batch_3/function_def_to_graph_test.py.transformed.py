
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function_def_to_graph
from tensorflow.python.framework import graph_to_function_def
from tensorflow.python.framework import op_def_library
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
class FunctionDefToGraphTest(test.TestCase):
  def _build_function_def(self):
    with ops.Graph().as_default() as g:
      x = array_ops.placeholder(dtypes.float32, name="x")
      y = array_ops.placeholder(dtypes.float32, name="y")
      sum_squares = math_ops.add_n(
          [math_ops.pow(x, 2), math_ops.pow(y, 2)], name="sum_squares")
      sum_cubes = math_ops.add_n(
          [math_ops.pow(x, 3), math_ops.pow(y, 3)], name="sum_cubes")
    fdef = graph_to_function_def.graph_to_function_def(
        g,
        g.get_operations(),
    fdef.signature.name = "_whats_in_a_name"
    return fdef
  @test_util.run_deprecated_v1
  def testInputsAndOutputs(self):
    fdef = self._build_function_def()
    g = function_def_to_graph.function_def_to_graph(fdef)
    self.assertEqual(g.name, "_whats_in_a_name")
    with self.session(graph=g) as sess:
      inputs = sess.run(g.inputs, feed_dict={"x:0": 2, "y:0": 3})
      self.assertSequenceEqual(inputs, [2.0, 3.0])
      outputs = sess.run(g.outputs, feed_dict={"x:0": 2, "y:0": 3})
      self.assertSequenceEqual(outputs, [13.0, 35.0])
  def testShapes(self):
    fdef = self._build_function_def()
    g = function_def_to_graph.function_def_to_graph(fdef)
    g = function_def_to_graph.function_def_to_graph(
        fdef,
        input_shapes=[
            tensor_shape.TensorShape([5]),
            tensor_shape.TensorShape([5])
        ])
    self.assertSequenceEqual(g.inputs[0].shape.dims, [5])
    self.assertSequenceEqual(g.inputs[1].shape.dims, [5])
    self.assertSequenceEqual(g.outputs[0].shape.dims, [5])
    self.assertSequenceEqual(g.outputs[1].shape.dims, [5])
    g = function_def_to_graph.function_def_to_graph(
        fdef, input_shapes=[None, tensor_shape.TensorShape([5, 7])])
    self.assertIsNone(g.inputs[0].shape.dims)
    self.assertSequenceEqual(g.inputs[1].shape.dims, [5, 7])
    self.assertSequenceEqual(g.outputs[0].shape.dims, [5, 7])
    self.assertSequenceEqual(g.outputs[1].shape.dims, [5, 7])
    with self.assertRaises(ValueError):
      g = function_def_to_graph.function_def_to_graph(
          fdef, input_shapes=[tensor_shape.TensorShape([5, 7])])
  def testResourceHandleInputShapes(self):
    with ops.Graph().as_default() as g:
      v = variables.Variable(array_ops.ones((2, 3), dtype=dtypes.float32))
      @def_function.function(
          input_signature=[tensor_spec.TensorSpec((None, 2, 2), dtypes.int32)])
      def lookup(inp):
        return {
            "shape inference": array_ops.gather_nd(v, inp),
            "handle": v.handle}
      lookup.get_concrete_function().add_to_graph()
      fdef = g.as_graph_def(add_shapes=True).library.function[0]
    fg = function_def_to_graph.function_def_to_graph(fdef)
    self.assertSequenceEqual(fg.inputs[0].shape.as_list(), [None, 2, 2])
    self.assertSequenceEqual(fg.inputs[1].shape.as_list(), [])
class FunctionDefToGraphDefTest(test.TestCase):
  def _build_function_def(self):
    with ops.Graph().as_default() as g:
      x = array_ops.placeholder(dtypes.float32, name="x")
      y = array_ops.placeholder(dtypes.int32, name="y")
      z = array_ops.placeholder(dtypes.int32, name="z")
      d_1, e_1 = op_def_library.apply_op("Foo1", name="foo_1", a=x, b=y, c=z)
      list_output0, list_output1 = test_ops.list_output(
          T=[dtypes.int32, dtypes.int32], name="list_output")
      d_2, e_2 = test_ops.foo1(a=d_1, b=e_1, c=list_output1, name="foo_2")
    fdef = graph_to_function_def.graph_to_function_def(
        g,
        g.get_operations(),
    assert fdef.node_def[0].op == "Foo1"
    assert fdef.node_def[0].input == ["x", "y", "z"]
    assert fdef.node_def[1].op == "ListOutput"
    assert not fdef.node_def[1].input
    assert fdef.node_def[2].op == "Foo1"
    assert fdef.node_def[2].input == [
        "foo_1:d:0", "foo_1:e:0", "list_output:a:1"
    ]
    return fdef
  def testTensorNames(self):
    fdef = self._build_function_def()
    g, tensor_name_map = function_def_to_graph.function_def_to_graph_def(fdef)
    self.assertSequenceEqual(g.node[3].input, ["x:0", "y:0", "z:0"])
    self.assertSequenceEqual(g.node[5].input,
                             ["foo_1:0", "foo_1:1", "list_output:1"])
    self.assertDictEqual(
        tensor_name_map, {
            "x": "x:0",
            "^x": "^x",
            "y": "y:0",
            "^y": "^y",
            "z": "z:0",
            "^z": "^z",
            "foo_1:d:0": "foo_1:0",
            "foo_1:e:0": "foo_1:1",
            "^foo_1": "^foo_1",
            "list_output:a:0": "list_output:0",
            "list_output:a:1": "list_output:1",
            "^list_output": "^list_output",
            "foo_2:d:0": "foo_2:0",
            "foo_2:e:0": "foo_2:1",
            "^foo_2": "^foo_2",
        })
  def testShapes(self):
    fdef = self._build_function_def()
    g, _ = function_def_to_graph.function_def_to_graph_def(
        fdef,
        input_shapes=[
            tensor_shape.TensorShape([]),
            tensor_shape.TensorShape([5]), None
        ])
    self.assertEqual("shape" in g.node[0].attr, True)
    self.assertSequenceEqual(
        tensor_shape.TensorShape(g.node[0].attr["shape"].shape).as_list(), [])
    self.assertEqual(g.node[0].attr["shape"].shape.unknown_rank, False)
    self.assertEqual("shape" in g.node[1].attr, True)
    self.assertSequenceEqual(
        tensor_shape.TensorShape(g.node[1].attr["shape"].shape).as_list(), [5])
    self.assertEqual(g.node[0].attr["shape"].shape.unknown_rank, False)
    self.assertFalse("shape" in g.node[2].attr)
  def testControlDependencies(self):
    v = variables.Variable(1)
    @function.defun
    def fn(inp):
      assign = v.assign(3, name="assign", read_value=False)
      x = constant_op.constant(2.0, name="x")
      with ops.control_dependencies([x, inp, assign]):
        constant_op.constant(3.0, name="y")
      return 4.0
    inp = constant_op.constant(1.0)
    fdef = fn.get_concrete_function(inp).function_def
    func_graph = function_def_to_graph.function_def_to_graph(fdef)
    op = func_graph.get_operation_by_name("y")
    self.assertEqual(len(op.control_inputs), 3)
    self.assertEqual(op.control_inputs[0].name, "assign")
    self.assertEqual(op.control_inputs[1].name, "inp")
    self.assertEqual(op.control_inputs[2].name, "x")
  def testAttributesForArgDef(self):
    @function.defun
    def fn(x):
      return x
    inp = constant_op.constant(1.0)
    fdef = fn.get_concrete_function(inp).function_def
    fdef.arg_attr[0].attr["_test_attr"].s = "value".encode("ascii")
    graph_def = function_def_to_graph.function_def_to_graph_def(fdef)
    placeholders = [
        ndef for ndef in graph_def[0].node if ndef.op == "Placeholder"
    ]
    self.assertEqual(1, len(placeholders))
    self.assertEqual(placeholders[0].attr["_test_attr"].s,
                     "value".encode("ascii"))
if __name__ == "__main__":
  test.main()
