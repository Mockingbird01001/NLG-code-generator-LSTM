
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.grappler import item as gitem
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
class PyWrapOptimizeGraphTest(test.TestCase):
  @test_util.run_deprecated_v1
  def testBasic(self):
    a = constant_op.constant(10, name='a')
    b = constant_op.constant(20, name='b')
    c = math_ops.add_n([a, b], name='c')
    d = math_ops.add_n([b, c], name='d')
    train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
    train_op.append(d)
    mg = meta_graph.create_meta_graph_def(graph=ops.get_default_graph())
    config = config_pb2.ConfigProto()
    rewriter_config = config.graph_options.rewrite_options
    rewriter_config.optimizers.append('constfold')
    rewriter_config.min_graph_nodes = -1
    graph = tf_optimizer.OptimizeGraph(config, mg)
    self.assertEqual(len(graph.node), 1)
    self.assertItemsEqual([node.name for node in graph.node], ['d'])
  @test_util.run_v1_only('b/120545219')
  def testKeepNodes(self):
    g = ops.Graph()
    with g.as_default():
      a1 = variables.VariableV1(
      a2 = constant_op.constant(0, shape=[50, 50], name='keep')
      with g._attr_scope(
          {'_grappler_do_not_remove': attr_value_pb2.AttrValue(b=True)}):
        a3 = constant_op.constant(0, name='keep2')
      b = constant_op.constant(1, shape=[100, 10])
      c = constant_op.constant(0, shape=[10, 30])
      d = math_ops.matmul(b, c)
    mg = meta_graph.create_meta_graph_def(graph=g)
    config = config_pb2.ConfigProto()
    rewriter_config = config.graph_options.rewrite_options
    rewriter_config.min_graph_nodes = -1
    optimized_graph = tf_optimizer.OptimizeGraph(config, mg)
    optimized_graph_nodes = [node.name for node in optimized_graph.node]
    expected_nodes = [
        d.op.name, a1.op.name, a2.op.name, a3.op.name, 'Variable/initial_value',
        'Variable/Assign'
    ]
    self.assertEqual(len(optimized_graph_nodes), len(expected_nodes))
    self.assertAllInSet(optimized_graph_nodes, expected_nodes)
  @test_util.run_v1_only('b/120545219')
  def testLoops(self):
    g = ops.Graph()
    with g.as_default():
      def _Cond(_, counter):
        return counter < end
      def _Body(buf, counter):
        buf = array_ops.concat([buf, [counter]], 0)
        counter += 1
        return [buf, counter]
      start = array_ops.placeholder(shape=[], dtype=dtypes.int32)
      end = array_ops.placeholder(shape=[], dtype=dtypes.int32)
      init_buf = array_ops.zeros(shape=[0], dtype=dtypes.int32)
      loop_vars = [init_buf, start]
      shape_inv = [
          tensor_shape.TensorShape([None]),
          tensor_shape.TensorShape([])
      ]
      buf, _ = control_flow_ops.while_loop(_Cond, _Body, loop_vars, shape_inv)
      buf_shape = array_ops.shape(buf)
      f_shape = array_ops.shape(f)
      ops.add_to_collection('train_op', buf_shape)
      ops.add_to_collection('train_op', f_shape)
    mg = meta_graph.create_meta_graph_def(graph=g)
    config = config_pb2.ConfigProto()
    rewriter_config = config.graph_options.rewrite_options
    rewriter_config.min_graph_nodes = -1
    optimized_graph = tf_optimizer.OptimizeGraph(config, mg)
    mg.graph_def.CopyFrom(optimized_graph)
    item = gitem.Item(mg)
    props = item.GetOpProperties()
    buf_prop = props[buf.op.name]
    f_prop = props[f.op.name]
    self.assertEqual(buf_prop, f_prop)
if __name__ == '__main__':
  test.main()
