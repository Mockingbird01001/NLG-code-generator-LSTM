
import re
import numpy as np
def assert_lines_equal_ignoring_whitespace(test, expected_lines, actual_lines):
  """Assert equality in lines, ignoring all whitespace.
  Args:
    test: An instance of unittest.TestCase or its subtypes (e.g.,
      TensorFlowTestCase).
    expected_lines: Expected lines as an iterable of strings.
    actual_lines: Actual lines as an iterable of strings.
  """
  test.assertEqual(
      len(expected_lines), len(actual_lines),
      "Mismatch in the number of lines: %d vs %d" % (
          len(expected_lines), len(actual_lines)))
  for expected_line, actual_line in zip(expected_lines, actual_lines):
    test.assertEqual("".join(expected_line.split()),
                     "".join(actual_line.split()))
_ARRAY_VALUE_SEPARATOR_REGEX = re.compile(r"(array|\(|\[|\]|\)|\||,)")
def assert_array_lines_close(test, expected_array, array_lines):
  """Assert that the array value represented by lines is close to expected.
  Note that the shape of the array represented by the `array_lines` is ignored.
  Args:
    test: An instance of TensorFlowTestCase.
    expected_array: Expected value of the array.
    array_lines: A list of strings representing the array.
      E.g., "array([[ 1.0, 2.0 ], [ 3.0, 4.0 ]])"
      Assumes that values are separated by commas, parentheses, brackets, "|"
      characters and whitespace.
  """
  elements = []
  for line in array_lines:
    line = re.sub(_ARRAY_VALUE_SEPARATOR_REGEX, " ", line)
    elements.extend(float(s) for s in line.split())
  test.assertAllClose(np.array(expected_array).flatten(), elements)
