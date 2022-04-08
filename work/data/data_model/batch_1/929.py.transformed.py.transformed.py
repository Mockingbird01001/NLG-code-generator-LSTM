
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import unittest
import six
class TextTestResult(unittest.TextTestResult):
  if six.PY2:
    def addSubTest(self, test, subtest, err):
      if err is not None:
        if getattr(self, 'failfast', False):
          self.stop()
        subtest_error_details = (subtest, self._exc_info_to_string(err, test))
        if issubclass(err[0], test.failureException):
          self.failures.append(subtest_error_details)
        else:
          self.errors.append(subtest_error_details)
        self._mirrorOutput = True
