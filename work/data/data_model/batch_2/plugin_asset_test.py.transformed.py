
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.summary import plugin_asset
class _UnnamedPluginAsset(plugin_asset.PluginAsset):
  def assets(self):
    return {}
class _ExamplePluginAsset(_UnnamedPluginAsset):
  plugin_name = "_ExamplePluginAsset"
class _OtherExampleAsset(_UnnamedPluginAsset):
  plugin_name = "_OtherExampleAsset"
class _ExamplePluginThatWillCauseCollision(_UnnamedPluginAsset):
  plugin_name = "_ExamplePluginAsset"
class PluginAssetTest(test_util.TensorFlowTestCase):
  def testGetPluginAsset(self):
    epa = plugin_asset.get_plugin_asset(_ExamplePluginAsset)
    self.assertIsInstance(epa, _ExamplePluginAsset)
    epa2 = plugin_asset.get_plugin_asset(_ExamplePluginAsset)
    self.assertIs(epa, epa2)
    opa = plugin_asset.get_plugin_asset(_OtherExampleAsset)
    self.assertIsNot(epa, opa)
  def testUnnamedPluginFails(self):
    with self.assertRaises(ValueError):
      plugin_asset.get_plugin_asset(_UnnamedPluginAsset)
  def testPluginCollisionDetected(self):
    plugin_asset.get_plugin_asset(_ExamplePluginAsset)
    with self.assertRaises(ValueError):
      plugin_asset.get_plugin_asset(_ExamplePluginThatWillCauseCollision)
  def testGetAllPluginAssets(self):
    epa = plugin_asset.get_plugin_asset(_ExamplePluginAsset)
    opa = plugin_asset.get_plugin_asset(_OtherExampleAsset)
    self.assertItemsEqual(plugin_asset.get_all_plugin_assets(), [epa, opa])
  def testRespectsGraphArgument(self):
    g1 = ops.Graph()
    g2 = ops.Graph()
    e1 = plugin_asset.get_plugin_asset(_ExamplePluginAsset, g1)
    e2 = plugin_asset.get_plugin_asset(_ExamplePluginAsset, g2)
    self.assertEqual(e1, plugin_asset.get_all_plugin_assets(g1)[0])
    self.assertEqual(e2, plugin_asset.get_all_plugin_assets(g2)[0])
if __name__ == "__main__":
  googletest.main()
