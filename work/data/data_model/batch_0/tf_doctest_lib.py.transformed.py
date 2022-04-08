
import doctest
import re
import textwrap
import numpy as np
class _FloatExtractor(object):
  """Class for extracting floats from a string.
  For example:
  >>> text_parts, floats = _FloatExtractor()("Text 1.0 Text")
  >>> text_parts
  ["Text ", " Text"]
  >>> floats
  np.array([1.0])
  """
  _FLOAT_RE = re.compile(
      r"""
        (?:
           )
        )
        )
      )
      )
      """.format(
          digits_dot_maybe_digits=r'(?:[0-9]+\.(?:[0-9]*))',
          dot_digits=r'(?:\.[0-9]+)',
          digits=r'(?:[0-9]+)',
          exponent=r'(?:[eE][-+]?[0-9]+)'),
      re.VERBOSE)
  def __call__(self, string):
    """Extracts floats from a string.
    >>> text_parts, floats = _FloatExtractor()("Text 1.0 Text")
    >>> text_parts
    ["Text ", " Text"]
    >>> floats
    np.array([1.0])
    Args:
      string: the string to extract floats from.
    Returns:
      A (string, array) pair, where `string` has each float replaced by "..."
      and `array` is a `float32` `numpy.array` containing the extracted floats.
    """
    texts = []
    floats = []
    for i, part in enumerate(self._FLOAT_RE.split(string)):
      if i % 2 == 0:
        texts.append(part)
      else:
        floats.append(float(part))
    return texts, np.array(floats)
class TfDoctestOutputChecker(doctest.OutputChecker, object):
  def __init__(self, *args, **kwargs):
    super(TfDoctestOutputChecker, self).__init__(*args, **kwargs)
    self.extract_floats = _FloatExtractor()
    self.text_good = None
    self.float_size_good = None
  _ADDRESS_RE = re.compile(r'\bat 0x[0-9a-f]*?>')
  _NUMPY_OUTPUT_RE = re.compile(r'<tf.Tensor.*?numpy=(.*?)>', re.DOTALL)
  def _allclose(self, want, got, rtol=1e-3, atol=1e-3):
    return np.allclose(want, got, rtol=rtol, atol=atol)
  def _tf_tensor_numpy_output(self, string):
    modified_string = self._NUMPY_OUTPUT_RE.sub(r'\1', string)
    return modified_string, modified_string != string
  MESSAGE = textwrap.dedent("""\n
        Check the documentation (https://www.tensorflow.org/community/contribute/docs_ref) on how to
        write testable docstrings.
  def check_output(self, want, got, optionflags):
    if got and not want:
      return True
    if want is None:
      want = ''
    want = self._ADDRESS_RE.sub('at ...>', want)
    want, want_changed = self._tf_tensor_numpy_output(want)
    if want_changed:
      got, _ = self._tf_tensor_numpy_output(got)
    want_text_parts, self.want_floats = self.extract_floats(want)
    want_text_wild = '...'.join(want_text_parts)
    _, self.got_floats = self.extract_floats(got)
    self.text_good = super(TfDoctestOutputChecker, self).check_output(
        want=want_text_wild, got=got, optionflags=optionflags)
    if not self.text_good:
      return False
    if self.want_floats.size == 0:
      return True
    self.float_size_good = (self.want_floats.size == self.got_floats.size)
    if self.float_size_good:
      return self._allclose(self.want_floats, self.got_floats)
    else:
      return False
  def output_difference(self, example, got, optionflags):
    got = [got]
    if self.text_good:
      if not self.float_size_good:
        got.append("\n\nCAUTION: tf_doctest doesn't work if *some* of the "
                   "*float output* is hidden with a \"...\".")
    got.append(self.MESSAGE)
    got = '\n'.join(got)
    return (super(TfDoctestOutputChecker,
                  self).output_difference(example, got, optionflags))
