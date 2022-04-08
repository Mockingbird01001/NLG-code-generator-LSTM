
from lib2to3.fixes.fix_imports import FixImports
from libfuturize.fixer_util import touch_import_top
class FixFutureStandardLibrary(FixImports):
    run_order = 8
    def transform(self, node, results):
        result = super(FixFutureStandardLibrary, self).transform(node, results)
        touch_import_top(u'future', u'standard_library', node)
        return result
