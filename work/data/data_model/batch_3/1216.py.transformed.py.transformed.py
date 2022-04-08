
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import datetime
import re
import sys
import threading
import time
import traceback
import unittest
from xml.sax import saxutils
from absl.testing import _pretty_print_reporter
from absl.third_party import unittest3_backport
import six
_bad_control_character_codes = set(range(0, 0x20)) - {0x9, 0xA, 0xD}
_control_character_conversions = {
    chr(i): '\\x{:02x}'.format(i) for i in _bad_control_character_codes}
_escape_xml_attr_conversions = {
    '"': '&quot;',
    "'": '&apos;',
_escape_xml_attr_conversions.update(_control_character_conversions)
_CLASS_OR_MODULE_LEVEL_TEST_DESC_REGEX = re.compile(r'^(\w+) \((\S+)\)$')
def _escape_xml_attr(content):
  return saxutils.escape(content, _escape_xml_attr_conversions)
def _escape_cdata(s):
  for char, escaped in six.iteritems(_control_character_conversions):
    s = s.replace(char, escaped)
  return s.replace(']]>', ']] >')
def _iso8601_timestamp(timestamp):
  if timestamp is None or timestamp < 0:
    return None
  if six.PY2:
    return '%s+00:00' % datetime.datetime.utcfromtimestamp(
        timestamp).isoformat()
  else:
    return datetime.datetime.fromtimestamp(
        timestamp, tz=datetime.timezone.utc).isoformat()
def _print_xml_element_header(element, attributes, stream, indentation=''):
  stream.write('%s<%s' % (indentation, element))
  for attribute in attributes:
    if len(attribute) == 2        and attribute[0] is not None and attribute[1] is not None:
      stream.write(' %s="%s"' % (attribute[0], attribute[1]))
  stream.write('>\n')
_time_copy = time.time
if hasattr(traceback, '_some_str'):
  _safe_str = traceback._some_str
else:
  _safe_str = str
class _TestCaseResult(object):
  def __init__(self, test):
    self.run_time = -1
    self.start_time = -1
    self.skip_reason = None
    self.errors = []
    self.test = test
    test_desc = test.id() or str(test)
    match = _CLASS_OR_MODULE_LEVEL_TEST_DESC_REGEX.match(test_desc)
    if match:
      name = match.group(1)
      full_class_name = match.group(2)
    else:
      class_name = unittest.util.strclass(test.__class__)
      if ((six.PY3 and isinstance(test, unittest.case._SubTest)) or
          (six.PY2 and isinstance(test, unittest3_backport.case._SubTest))):
        class_name = unittest.util.strclass(test.test_case.__class__)
      if test_desc.startswith(class_name + '.'):
        name = test_desc[len(class_name)+1:]
        full_class_name = class_name
      else:
        parts = test_desc.rsplit('.', 1)
        name = parts[-1]
        full_class_name = parts[0] if len(parts) == 2 else ''
    self.name = _escape_xml_attr(name)
    self.full_class_name = _escape_xml_attr(full_class_name)
  def set_run_time(self, time_in_secs):
    self.run_time = time_in_secs
  def set_start_time(self, time_in_secs):
    self.start_time = time_in_secs
  def print_xml_summary(self, stream):
    if self.skip_reason is None:
      status = 'run'
      result = 'completed'
    else:
      status = 'notrun'
      result = 'suppressed'
    test_case_attributes = [
        ('name', '%s' % self.name),
        ('status', '%s' % status),
        ('result', '%s' % result),
        ('time', '%.1f' % self.run_time),
        ('classname', self.full_class_name),
        ('timestamp', _iso8601_timestamp(self.start_time)),
    ]
    _print_xml_element_header('testcase', test_case_attributes, stream, '  ')
    self._print_testcase_details(stream)
    stream.write('  </testcase>\n')
  def _print_testcase_details(self, stream):
    for error in self.errors:
      outcome, exception_type, message, error_msg = error
      message = _escape_xml_attr(_safe_str(message))
      exception_type = _escape_xml_attr(str(exception_type))
      error_msg = _escape_cdata(error_msg)
      stream.write('  <%s message="%s" type="%s"><![CDATA[%s]]></%s>\n'
                   % (outcome, message, exception_type, error_msg, outcome))
class _TestSuiteResult(object):
  def __init__(self):
    self.suites = {}
    self.failure_counts = {}
    self.error_counts = {}
    self.overall_start_time = -1
    self.overall_end_time = -1
    self._testsuites_properties = {}
  def add_test_case_result(self, test_case_result):
    suite_name = type(test_case_result.test).__name__
    if suite_name == '_ErrorHolder':
      suite_name = test_case_result.full_class_name.rsplit('.')[-1]
    if ((six.PY3 and
         isinstance(test_case_result.test, unittest.case._SubTest)) or
        (six.PY2 and
         isinstance(test_case_result.test, unittest3_backport.case._SubTest))):
      suite_name = type(test_case_result.test.test_case).__name__
    self._setup_test_suite(suite_name)
    self.suites[suite_name].append(test_case_result)
    for error in test_case_result.errors:
      if error[0] == 'failure':
        self.failure_counts[suite_name] += 1
        break
      elif error[0] == 'error':
        self.error_counts[suite_name] += 1
        break
  def print_xml_summary(self, stream):
    overall_test_count = sum(len(x) for x in self.suites.values())
    overall_failures = sum(self.failure_counts.values())
    overall_errors = sum(self.error_counts.values())
    overall_attributes = [
        ('name', ''),
        ('tests', '%d' % overall_test_count),
        ('failures', '%d' % overall_failures),
        ('errors', '%d' % overall_errors),
        ('time', '%.1f' % (self.overall_end_time - self.overall_start_time)),
        ('timestamp', _iso8601_timestamp(self.overall_start_time)),
    ]
    _print_xml_element_header('testsuites', overall_attributes, stream)
    if self._testsuites_properties:
      stream.write('    <properties>\n')
      for name, value in sorted(six.iteritems(self._testsuites_properties)):
        stream.write('      <property name="%s" value="%s"></property>\n' %
                     (_escape_xml_attr(name), _escape_xml_attr(str(value))))
      stream.write('    </properties>\n')
    for suite_name in self.suites:
      suite = self.suites[suite_name]
      suite_end_time = max(x.start_time + x.run_time for x in suite)
      suite_start_time = min(x.start_time for x in suite)
      failures = self.failure_counts[suite_name]
      errors = self.error_counts[suite_name]
      suite_attributes = [
          ('name', '%s' % suite_name),
          ('tests', '%d' % len(suite)),
          ('failures', '%d' % failures),
          ('errors', '%d' % errors),
          ('time', '%.1f' % (suite_end_time - suite_start_time)),
          ('timestamp', _iso8601_timestamp(suite_start_time)),
      ]
      _print_xml_element_header('testsuite', suite_attributes, stream)
      for test_case_result in suite:
        test_case_result.print_xml_summary(stream)
      stream.write('</testsuite>\n')
    stream.write('</testsuites>\n')
  def _setup_test_suite(self, suite_name):
    if suite_name in self.suites:
      return
    self.suites[suite_name] = []
    self.failure_counts[suite_name] = 0
    self.error_counts[suite_name] = 0
  def set_end_time(self, timestamp_in_secs):
    self.overall_end_time = timestamp_in_secs
  def set_start_time(self, timestamp_in_secs):
    self.overall_start_time = timestamp_in_secs
class _TextAndXMLTestResult(_pretty_print_reporter.TextTestResult):
  _TEST_SUITE_RESULT_CLASS = _TestSuiteResult
  _TEST_CASE_RESULT_CLASS = _TestCaseResult
  def __init__(self, xml_stream, stream, descriptions, verbosity,
               time_getter=_time_copy, testsuites_properties=None):
    super(_TextAndXMLTestResult, self).__init__(stream, descriptions, verbosity)
    self.xml_stream = xml_stream
    self.pending_test_case_results = {}
    self.suite = self._TEST_SUITE_RESULT_CLASS()
    if testsuites_properties:
      self.suite._testsuites_properties = testsuites_properties
    self.time_getter = time_getter
    self._pending_test_case_results_lock = threading.RLock()
  def startTest(self, test):
    self.start_time = self.time_getter()
    super(_TextAndXMLTestResult, self).startTest(test)
  def stopTest(self, test):
    with self._pending_test_case_results_lock:
      super(_TextAndXMLTestResult, self).stopTest(test)
      result = self.get_pending_test_case_result(test)
      if not result:
        test_name = test.id() or str(test)
        sys.stderr.write('No pending test case: %s\n' % test_name)
        return
      test_id = id(test)
      run_time = self.time_getter() - self.start_time
      result.set_run_time(run_time)
      result.set_start_time(self.start_time)
      self.suite.add_test_case_result(result)
      del self.pending_test_case_results[test_id]
  def startTestRun(self):
    self.suite.set_start_time(self.time_getter())
    super(_TextAndXMLTestResult, self).startTestRun()
  def stopTestRun(self):
    self.suite.set_end_time(self.time_getter())
    with self._pending_test_case_results_lock:
      for test_id in self.pending_test_case_results:
        result = self.pending_test_case_results[test_id]
        if hasattr(self, 'start_time'):
          run_time = self.suite.overall_end_time - self.start_time
          result.set_run_time(run_time)
          result.set_start_time(self.start_time)
        self.suite.add_test_case_result(result)
      self.pending_test_case_results.clear()
  def _exc_info_to_string(self, err, test=None):
    if test:
      return super(_TextAndXMLTestResult, self)._exc_info_to_string(err, test)
    return ''.join(traceback.format_exception(*err))
  def add_pending_test_case_result(self, test, error_summary=None,
                                   skip_reason=None):
    with self._pending_test_case_results_lock:
      test_id = id(test)
      if test_id not in self.pending_test_case_results:
        self.pending_test_case_results[test_id] = self._TEST_CASE_RESULT_CLASS(
            test)
      if error_summary:
        self.pending_test_case_results[test_id].errors.append(error_summary)
      if skip_reason:
        self.pending_test_case_results[test_id].skip_reason = skip_reason
  def delete_pending_test_case_result(self, test):
    with self._pending_test_case_results_lock:
      test_id = id(test)
      del self.pending_test_case_results[test_id]
  def get_pending_test_case_result(self, test):
    test_id = id(test)
    return self.pending_test_case_results.get(test_id, None)
  def addSuccess(self, test):
    super(_TextAndXMLTestResult, self).addSuccess(test)
    self.add_pending_test_case_result(test)
  def addError(self, test, err):
    super(_TextAndXMLTestResult, self).addError(test, err)
    error_summary = ('error', err[0], err[1],
                     self._exc_info_to_string(err, test=test))
    self.add_pending_test_case_result(test, error_summary=error_summary)
  def addFailure(self, test, err):
    super(_TextAndXMLTestResult, self).addFailure(test, err)
    error_summary = ('failure', err[0], err[1],
                     self._exc_info_to_string(err, test=test))
    self.add_pending_test_case_result(test, error_summary=error_summary)
  def addSkip(self, test, reason):
    super(_TextAndXMLTestResult, self).addSkip(test, reason)
    self.add_pending_test_case_result(test, skip_reason=reason)
  def addExpectedFailure(self, test, err):
    super(_TextAndXMLTestResult, self).addExpectedFailure(test, err)
    if callable(getattr(test, 'recordProperty', None)):
      test.recordProperty('EXPECTED_FAILURE',
                          self._exc_info_to_string(err, test=test))
    self.add_pending_test_case_result(test)
  def addUnexpectedSuccess(self, test):
    super(_TextAndXMLTestResult, self).addUnexpectedSuccess(test)
    test_name = test.id() or str(test)
    error_summary = ('error', '', '',
                     'Test case %s should have failed, but passed.'
                     % (test_name))
    self.add_pending_test_case_result(test, error_summary=error_summary)
  def addSubTest(self, test, subtest, err):
    super(_TextAndXMLTestResult, self).addSubTest(test, subtest, err)
    if err is not None:
      if issubclass(err[0], test.failureException):
        error_summary = ('failure', err[0], err[1],
                         self._exc_info_to_string(err, test=test))
      else:
        error_summary = ('error', err[0], err[1],
                         self._exc_info_to_string(err, test=test))
    else:
      error_summary = None
    self.add_pending_test_case_result(subtest, error_summary=error_summary)
  def printErrors(self):
    super(_TextAndXMLTestResult, self).printErrors()
    self.xml_stream.write('<?xml version="1.0"?>\n')
    self.suite.print_xml_summary(self.xml_stream)
class TextAndXMLTestRunner(unittest.TextTestRunner):
  _TEST_RESULT_CLASS = _TextAndXMLTestResult
  _xml_stream = None
  _testsuites_properties = {}
  def __init__(self, xml_stream=None, *args, **kwargs):
    super(TextAndXMLTestRunner, self).__init__(*args, **kwargs)
    if xml_stream is not None:
      self._xml_stream = xml_stream
  @classmethod
  def set_default_xml_stream(cls, xml_stream):
    cls._xml_stream = xml_stream
  def _makeResult(self):
    if self._xml_stream is None:
      return super(TextAndXMLTestRunner, self)._makeResult()
    else:
      return self._TEST_RESULT_CLASS(
          self._xml_stream, self.stream, self.descriptions, self.verbosity,
          testsuites_properties=self._testsuites_properties)
  @classmethod
  def set_testsuites_property(cls, key, value):
    cls._testsuites_properties[key] = value
