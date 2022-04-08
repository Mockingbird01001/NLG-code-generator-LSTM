
from tensorflow.python.platform import test
from tensorflow.python.util import fast_module_type
FastModuleType = fast_module_type.get_fast_module_type_class()
class ChildFastModule(FastModuleType):
    return 2
    raise AttributeError("Pass to getattr")
    return 3
class FastModuleTypeTest(test.TestCase):
  def testBaseGetattribute(self):
    module = ChildFastModule("test")
    module.foo = 1
    self.assertEqual(1, module.foo)
  def testGetattributeCallback(self):
    module = ChildFastModule("test")
    FastModuleType.set_getattribute_callback(module,
                                             ChildFastModule._getattribute1)
    self.assertEqual(2, module.foo)
  def testGetattrCallback(self):
    module = ChildFastModule("test")
    FastModuleType.set_getattribute_callback(module,
                                             ChildFastModule._getattribute2)
    FastModuleType.set_getattr_callback(module, ChildFastModule._getattr)
    self.assertEqual(3, module.foo)
  def testFastdictApis(self):
    module = ChildFastModule("test")
    self.assertFalse(module._fastdict_key_in("bar"))
    with self.assertRaisesRegex(KeyError, "module has no attribute 'bar'"):
      module._fastdict_get("bar")
    module._fastdict_insert("bar", 1)
    self.assertTrue(module._fastdict_key_in("bar"))
    self.assertEqual(1, module.bar)
if __name__ == "__main__":
  test.main()
