
import os
import sys
import doctest
import inspect
import numpy
import nose
from nose.plugins import doctests as npd
from nose.plugins.errorclass import ErrorClass, ErrorClassPlugin
from nose.plugins.base import Plugin
from nose.util import src
from .nosetester import get_package_name
from .utils import KnownFailureException, KnownFailureTest
class NumpyDocTestFinder(doctest.DocTestFinder):
    def _from_module(self, module, object):
        if module is None:
            return True
        elif inspect.isfunction(object):
            return module.__dict__ is object.__globals__
        elif inspect.isbuiltin(object):
            return module.__name__ == object.__module__
        elif inspect.isclass(object):
            return module.__name__ == object.__module__
        elif inspect.ismethod(object):
            return module.__name__ == object.__self__.__class__.__module__
        elif inspect.getmodule(object) is not None:
            return module is inspect.getmodule(object)
        elif hasattr(object, '__module__'):
            return module.__name__ == object.__module__
        elif isinstance(object, property):
            return True
        else:
            raise ValueError("object must be a class or function")
    def _find(self, tests, obj, name, module, source_lines, globs, seen):
        doctest.DocTestFinder._find(self, tests, obj, name, module,
                                    source_lines, globs, seen)
        from inspect import (
            isroutine, isclass, ismodule, isfunction, ismethod
            )
        if ismodule(obj) and self._recurse:
            for valname, val in obj.__dict__.items():
                valname1 = f'{name}.{valname}'
                if ( (isroutine(val) or isclass(val))
                     and self._from_module(module, val)):
                    self._find(tests, val, valname1, module, source_lines,
                               globs, seen)
        if isclass(obj) and self._recurse:
            for valname, val in obj.__dict__.items():
                if isinstance(val, staticmethod):
                    val = getattr(obj, valname)
                if isinstance(val, classmethod):
                    val = getattr(obj, valname).__func__
                if ((isfunction(val) or isclass(val) or
                     ismethod(val) or isinstance(val, property)) and
                      self._from_module(module, val)):
                    valname = f'{name}.{valname}'
                    self._find(tests, val, valname, module, source_lines,
                               globs, seen)
class NumpyOutputChecker(doctest.OutputChecker):
    def check_output(self, want, got, optionflags):
        ret = doctest.OutputChecker.check_output(self, want, got,
                                                 optionflags)
        if not ret:
                return True
            got = got.replace("'>", "'<")
            want = want.replace("'>", "'<")
            for sz in [4, 8]:
                got = got.replace("'<i%d'" % sz, "int")
                want = want.replace("'<i%d'" % sz, "int")
            ret = doctest.OutputChecker.check_output(self, want,
                    got, optionflags)
        return ret
class NumpyDocTestCase(npd.DocTestCase):
    def __init__(self, test, optionflags=0, setUp=None, tearDown=None,
                 checker=None, obj=None, result_var='_'):
        self._result_var = result_var
        self._nose_obj = obj
        doctest.DocTestCase.__init__(self, test,
                                     optionflags=optionflags,
                                     setUp=setUp, tearDown=tearDown,
                                     checker=checker)
print_state = numpy.get_printoptions()
class NumpyDoctest(npd.Doctest):
    name = 'numpydoctest'
    score = 1000
    doctest_optflags = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
    doctest_ignore = ['generate_numpy_api.py',
                      'setup.py']
    doctest_case_class = NumpyDocTestCase
    out_check_class = NumpyOutputChecker
    test_finder_class = NumpyDocTestFinder
    def options(self, parser, env=os.environ):
        Plugin.options(self, parser, env)
        self.doctest_tests = True
        self.doctest_result_var = None
    def configure(self, options, config):
        Plugin.configure(self, options, config)
        self.finder = self.test_finder_class()
        self.parser = doctest.DocTestParser()
        if self.enabled:
            config.plugins.plugins = [p for p in config.plugins.plugins
                                      if p.name != 'doctest']
    def set_test_context(self, test):
        pkg_name = get_package_name(os.path.dirname(test.filename))
        test.globs = {'__builtins__':__builtins__,
                      '__file__':'__main__',
                      '__name__':'__main__',
                      'np':numpy}
        if 'scipy' in pkg_name:
            p = pkg_name.split('.')
            p2 = p[-1]
            test.globs[p2] = __import__(pkg_name, test.globs, {}, [p2])
    def loadTestsFromModule(self, module):
        if not self.matches(module.__name__):
            npd.log.debug("Doctest doesn't want module %s", module)
            return
        try:
            tests = self.finder.find(module)
        except AttributeError:
            return
        if not tests:
            return
        tests.sort()
        module_file = src(module.__file__)
        for test in tests:
            if not test.examples:
                continue
            if not test.filename:
                test.filename = module_file
            self.set_test_context(test)
            yield self.doctest_case_class(test,
                                          optionflags=self.doctest_optflags,
                                          checker=self.out_check_class(),
                                          result_var=self.doctest_result_var)
    def afterContext(self):
        numpy.set_printoptions(**print_state)
    def wantFile(self, file):
        bn = os.path.basename(file)
        if bn in self.doctest_ignore:
            return False
        return npd.Doctest.wantFile(self, file)
class Unplugger:
    name = 'unplugger'
    enabled = True
    score = 4000
    def __init__(self, to_unplug='doctest'):
        self.to_unplug = to_unplug
    def options(self, parser, env):
        pass
    def configure(self, options, config):
        config.plugins.plugins = [p for p in config.plugins.plugins
                                  if p.name != self.to_unplug]
class KnownFailurePlugin(ErrorClassPlugin):
    enabled = True
    knownfail = ErrorClass(KnownFailureException,
                           label='KNOWNFAIL',
                           isfailure=False)
    def options(self, parser, env=os.environ):
        env_opt = 'NOSE_WITHOUT_KNOWNFAIL'
        parser.add_option('--no-knownfail', action='store_true',
                          dest='noKnownFail', default=env.get(env_opt, False),
                          help='Disable special handling of KnownFailure '
                               'exceptions')
    def configure(self, options, conf):
        if not self.can_configure:
            return
        self.conf = conf
        disable = getattr(options, 'noKnownFail', False)
        if disable:
            self.enabled = False
KnownFailure = KnownFailurePlugin
class FPUModeCheckPlugin(Plugin):
    def prepareTestCase(self, test):
        from numpy.core._multiarray_tests import get_fpu_mode
        def run(result):
            old_mode = get_fpu_mode()
            test.test(result)
            new_mode = get_fpu_mode()
            if old_mode != new_mode:
                try:
                    raise AssertionError(
                        "test".format(old_mode, new_mode))
                except AssertionError:
                    result.addFailure(test, sys.exc_info())
        return run
class NumpyTestProgram(nose.core.TestProgram):
    def runTests(self):
        if self.testRunner is None:
            self.testRunner = nose.core.TextTestRunner(stream=self.config.stream,
                                                       verbosity=self.config.verbosity,
                                                       config=self.config)
        plug_runner = self.config.plugins.prepareTestRunner(self.testRunner)
        if plug_runner is not None:
            self.testRunner = plug_runner
        self.result = self.testRunner.run(self.test)
        self.success = self.result.wasSuccessful()
        return self.success
