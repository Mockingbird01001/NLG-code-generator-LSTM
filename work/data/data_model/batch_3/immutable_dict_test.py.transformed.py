
from absl.testing import parameterized
from tensorflow.python.framework import immutable_dict
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
class ImmutableDictTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  def testGetItem(self):
    d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
    self.assertEqual(d['x'], 1)
    self.assertEqual(d['y'], 2)
    with self.assertRaises(KeyError):
  def testIter(self):
    d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
    self.assertEqual(set(iter(d)), set(['x', 'y']))
  def testContains(self):
    d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
    self.assertIn('x', d)
    self.assertIn('y', d)
    self.assertNotIn('z', d)
  def testLen(self):
    d1 = immutable_dict.ImmutableDict({})
    d2 = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
    self.assertLen(d2, 2)
  def testRepr(self):
    d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
    s = repr(d)
    self.assertTrue(s == "ImmutableDict({'x': 1, 'y': 2})" or
                    s == "ImmutableDict({'y': 1, 'x': 2})")
  def testGet(self):
    d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
    self.assertEqual(d.get('x'), 1)
    self.assertEqual(d.get('y'), 2)
    self.assertIsNone(d.get('z'))
    self.assertEqual(d.get('z', 'Foo'), 'Foo')
  def testKeys(self):
    d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
    self.assertEqual(set(d.keys()), set(['x', 'y']))
  def testValues(self):
    d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
    self.assertEqual(set(d.values()), set([1, 2]))
  def testItems(self):
    d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
    self.assertEqual(set(d.items()), set([('x', 1), ('y', 2)]))
  def testEqual(self):
    d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
    self.assertEqual(d, {'x': 1, 'y': 2})
  def testNotEqual(self):
    d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
    self.assertNotEqual(d, {'x': 1})
  def testSetItemFails(self):
    d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
    with self.assertRaises(TypeError):
    with self.assertRaises(TypeError):
  def testDelItemFails(self):
    d = immutable_dict.ImmutableDict({'x': 1, 'y': 2})
    with self.assertRaises(TypeError):
    with self.assertRaises(TypeError):
if __name__ == '__main__':
  googletest.main()
