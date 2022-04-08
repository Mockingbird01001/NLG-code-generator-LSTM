
"""Module to convert log levels between Abseil Python, C++, and Python standard.
This converter has to convert (best effort) between three different
logging level schemes:
  cpp      = The C++ logging level scheme used in Abseil C++.
  absl     = The absl.logging level scheme used in Abseil Python.
  standard = The python standard library logging level scheme.
Here is a handy ascii chart for easy mental mapping.
  LEVEL    | cpp |  absl  | standard |
  ---------+-----+--------+----------+
  DEBUG    |  0  |    1   |    10    |
  INFO     |  0  |    0   |    20    |
  WARNING  |  1  |   -1   |    30    |
  ERROR    |  2  |   -2   |    40    |
  CRITICAL |  3  |   -3   |    50    |
  FATAL    |  3  |   -3   |    50    |
Note: standard logging CRITICAL is mapped to absl/cpp FATAL.
However, only CRITICAL logs from the absl logger (or absl.logging.fatal) will
terminate the program. CRITICAL logs from non-absl loggers are treated as
error logs with a message prefix "CRITICAL - ".
Converting from standard to absl or cpp is a lossy conversion.
Converting back to standard will lose granularity.  For this reason,
users should always try to convert to standard, the richest
representation, before manipulating the levels, and then only to cpp
or absl if those level schemes are absolutely necessary.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
STANDARD_CRITICAL = logging.CRITICAL
STANDARD_ERROR = logging.ERROR
STANDARD_WARNING = logging.WARNING
STANDARD_INFO = logging.INFO
STANDARD_DEBUG = logging.DEBUG
ABSL_FATAL = -3
ABSL_ERROR = -2
ABSL_WARNING = -1
ABSL_WARN = -1
ABSL_INFO = 0
ABSL_DEBUG = 1
ABSL_LEVELS = {ABSL_FATAL: 'FATAL',
               ABSL_ERROR: 'ERROR',
               ABSL_WARNING: 'WARNING',
               ABSL_INFO: 'INFO',
               ABSL_DEBUG: 'DEBUG'}
ABSL_NAMES = {'FATAL': ABSL_FATAL,
              'ERROR': ABSL_ERROR,
              'WARNING': ABSL_WARNING,
              'WARN': ABSL_WARNING,
              'INFO': ABSL_INFO,
              'DEBUG': ABSL_DEBUG}
ABSL_TO_STANDARD = {ABSL_FATAL: STANDARD_CRITICAL,
                    ABSL_ERROR: STANDARD_ERROR,
                    ABSL_WARNING: STANDARD_WARNING,
                    ABSL_INFO: STANDARD_INFO,
                    ABSL_DEBUG: STANDARD_DEBUG}
STANDARD_TO_ABSL = dict((v, k) for (k, v) in ABSL_TO_STANDARD.items())
def get_initial_for_level(level):
  if level < STANDARD_WARNING:
    return 'I'
  elif level < STANDARD_ERROR:
    return 'W'
  elif level < STANDARD_CRITICAL:
    return 'E'
  else:
    return 'F'
def absl_to_cpp(level):
  if not isinstance(level, int):
    raise TypeError('Expect an int level, found {}'.format(type(level)))
  if level >= 0:
    return 0
  else:
    return -level
def absl_to_standard(level):
  if not isinstance(level, int):
    raise TypeError('Expect an int level, found {}'.format(type(level)))
  if level < ABSL_FATAL:
    level = ABSL_FATAL
  if level <= ABSL_DEBUG:
    return ABSL_TO_STANDARD[level]
  return STANDARD_DEBUG - level + 1
def string_to_standard(level):
  return absl_to_standard(ABSL_NAMES.get(level.upper()))
def standard_to_absl(level):
  if not isinstance(level, int):
    raise TypeError('Expect an int level, found {}'.format(type(level)))
  if level < 0:
    level = 0
  if level < STANDARD_DEBUG:
    return STANDARD_DEBUG - level + 1
  elif level < STANDARD_INFO:
    return ABSL_DEBUG
  elif level < STANDARD_WARNING:
    return ABSL_INFO
  elif level < STANDARD_ERROR:
    return ABSL_WARNING
  elif level < STANDARD_CRITICAL:
    return ABSL_ERROR
  else:
    return ABSL_FATAL
def standard_to_cpp(level):
  return absl_to_cpp(standard_to_absl(level))
