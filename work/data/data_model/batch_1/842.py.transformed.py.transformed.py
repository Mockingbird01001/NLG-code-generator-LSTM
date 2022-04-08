
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import unittest
from absl.third_party import unittest3_backport
class TextTestResult(unittest3_backport.TextTestResult):
  def __init__(self, stream, descriptions, verbosity):
    super(TextTestResult, self).__init__(stream, descriptions, 0)
    self._per_test_output = verbosity > 0
  def _print_status(self, tag, test):
    if self._per_test_output:
      test_id = test.id()
      if test_id.startswith('__main__.'):
        test_id = test_id[len('__main__.'):]
      print('[%s] %s' % (tag, test_id), file=self.stream)
      self.stream.flush()
  def startTest(self, test):
    super(TextTestResult, self).startTest(test)
    self._print_status(' RUN      ', test)
  def addSuccess(self, test):
    super(TextTestResult, self).addSuccess(test)
    self._print_status('       OK ', test)
  def addError(self, test, err):
    super(TextTestResult, self).addError(test, err)
    self._print_status('  FAILED  ', test)
  def addFailure(self, test, err):
    super(TextTestResult, self).addFailure(test, err)
    self._print_status('  FAILED  ', test)
  def addSkip(self, test, reason):
    super(TextTestResult, self).addSkip(test, reason)
    self._print_status('  SKIPPED ', test)
  def addExpectedFailure(self, test, err):
    super(TextTestResult, self).addExpectedFailure(test, err)
    self._print_status('       OK ', test)
  def addUnexpectedSuccess(self, test):
    super(TextTestResult, self).addUnexpectedSuccess(test)
    self._print_status('  FAILED  ', test)
class TextTestRunner(unittest.TextTestRunner):
  _TEST_RESULT_CLASS = TextTestResult
  run_for_debugging = False
  def run(self, test):
    if self.run_for_debugging:
      return self._run_debug(test)
    else:
      return super(TextTestRunner, self).run(test)
  def _run_debug(self, test):
    test.debug()
    return self._makeResult()
  def _makeResult(self):
    return TextTestResult(self.stream, self.descriptions, self.verbosity)
