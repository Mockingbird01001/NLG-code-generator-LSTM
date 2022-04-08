
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
class ApiDefMapTest(test_util.TensorFlowTestCase):
  def testApiDefMapOpNames(self):
    api_def_map = c_api_util.ApiDefMap()
    self.assertIn("Add", api_def_map.op_names())
  def testApiDefMapGet(self):
    api_def_map = c_api_util.ApiDefMap()
    op_def = api_def_map.get_op_def("Add")
    self.assertEqual(op_def.name, "Add")
    api_def = api_def_map.get_api_def("Add")
    self.assertEqual(api_def.graph_op_name, "Add")
  def testApiDefMapPutThenGet(self):
    api_def_map = c_api_util.ApiDefMap()
    api_def_text = """
op {
  graph_op_name: "Add"
  summary: "Returns x + y element-wise."
  description: <<END
*NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
END
}
"""
    api_def_map.put_api_def(api_def_text)
    api_def = api_def_map.get_api_def("Add")
    self.assertEqual(api_def.graph_op_name, "Add")
    self.assertEqual(api_def.summary, "Returns x + y element-wise.")
if __name__ == "__main__":
  googletest.main()
