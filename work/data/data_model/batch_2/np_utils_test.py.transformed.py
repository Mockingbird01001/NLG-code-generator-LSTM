
from absl.testing import parameterized
from tensorflow.python.ops.numpy_ops import np_utils
from tensorflow.python.platform import test
class UtilsTest(test.TestCase, parameterized.TestCase):
  def setUp(self):
    super(UtilsTest, self).setUp()
    self._old_np_doc_form = np_utils.get_np_doc_form()
    self._old_is_sig_mismatch_an_error = np_utils.is_sig_mismatch_an_error()
  def tearDown(self):
    np_utils.set_np_doc_form(self._old_np_doc_form)
    np_utils.set_is_sig_mismatch_an_error(self._old_is_sig_mismatch_an_error)
    super(UtilsTest, self).tearDown()
  def testNpDocInlined(self):
    def np_fun(x, y, z):
      return
    np_utils.set_np_doc_form('inlined')
    @np_utils.np_doc(None, np_fun=np_fun, unsupported_params=['x'])
    def f(x, z):
      return
    expected =
    self.assertEqual(expected, f.__doc__)
  @parameterized.named_parameters([
      [('dev',
        'https://numpy.org/devdocs/reference/generated/numpy.np_fun.html'),
       ('stable',
        'https://numpy.org/doc/stable/reference/generated/numpy.np_fun.html'),
       ('1.16',
        'https://numpy.org/doc/1.16/reference/generated/numpy.np_fun.html')
      ]])
  def testNpDocLink(self, version, link):
    def np_fun(x, y, z):
      return
    np_utils.set_np_doc_form(version)
    @np_utils.np_doc(None, np_fun=np_fun, unsupported_params=['x'])
    def f(x, z):
      return
    expected = """TensorFlow variant of NumPy's `np_fun`.
Unsupported arguments: `x`, `y`.
f docstring.
See the NumPy documentation for [`numpy.np_fun`](%s)."""
    expected = expected % (link)
    self.assertEqual(expected, f.__doc__)
  @parameterized.parameters([None, 1, 'a', '1a', '1.1a', '1.1.1a'])
  def testNpDocInvalid(self, invalid_flag):
    def np_fun(x, y, z):
      return
    np_utils.set_np_doc_form(invalid_flag)
    @np_utils.np_doc(None, np_fun=np_fun, unsupported_params=['x'])
    def f(x, z):
      return
    expected =
    self.assertEqual(expected, f.__doc__)
  def testNpDocName(self):
    np_utils.set_np_doc_form('inlined')
    @np_utils.np_doc('foo')
    def f():
      return
    expected =
    self.assertEqual(expected, f.__doc__)
  def testSigMismatchIsError(self):
    if not np_utils._supports_signature():
      self.skipTest('inspect.signature not supported')
    np_utils.set_is_sig_mismatch_an_error(True)
    def np_fun(x, y=1, **kwargs):
      return
    with self.assertRaisesRegex(TypeError, 'Cannot find parameter'):
      @np_utils.np_doc(None, np_fun=np_fun)
      def f1(a):
        return
    with self.assertRaisesRegex(TypeError, 'is of kind'):
      @np_utils.np_doc(None, np_fun=np_fun)
      def f2(x, kwargs):
        return
    with self.assertRaisesRegex(
        TypeError, 'Parameter y should have a default value'):
      @np_utils.np_doc(None, np_fun=np_fun)
      def f3(x, y):
        return
  def testSigMismatchIsNotError(self):
    np_utils.set_is_sig_mismatch_an_error(False)
    def np_fun(x, y=1, **kwargs):
      return
    @np_utils.np_doc(None, np_fun=np_fun)
    def f1(a):
      return
    def f2(x, kwargs):
      return
    @np_utils.np_doc(None, np_fun=np_fun)
    def f3(x, y):
      return
if __name__ == '__main__':
  test.main()
