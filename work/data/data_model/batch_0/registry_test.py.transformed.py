
from absl.testing import parameterized
from tensorflow.python.framework import registry
from tensorflow.python.platform import test
def bar():
  pass
class RegistryTest(test.TestCase, parameterized.TestCase):
  class Foo(object):
    pass
  @parameterized.parameters([Foo, bar])
  def testRegistryBasics(self, candidate):
    myreg = registry.Registry('testRegistry')
    with self.assertRaises(LookupError):
      myreg.lookup('testKey')
    myreg.register(candidate)
    self.assertEqual(myreg.lookup(candidate.__name__), candidate)
    myreg.register(candidate, 'testKey')
    self.assertEqual(myreg.lookup('testKey'), candidate)
    self.assertEqual(
        sorted(myreg.list()), sorted(['testKey', candidate.__name__]))
  def testDuplicate(self):
    myreg = registry.Registry('testbar')
    myreg.register(bar, 'Bar')
    with self.assertRaisesRegex(
        KeyError, r'Registering two testbar with name \'Bar\'! '
        r'\(Previous registration was in [^ ]+ .*.py:[0-9]+\)'):
      myreg.register(bar, 'Bar')
if __name__ == '__main__':
  test.main()
