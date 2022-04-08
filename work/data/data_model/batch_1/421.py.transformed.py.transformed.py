
from lib2to3.fixes.fix_urllib import FixUrllib
from libfuturize.fixer_util import touch_import_top, find_root
class FixFutureStandardLibraryUrllib(FixUrllib):
    run_order = 8
    def transform(self, node, results):
        root = find_root(node)
        result = super(FixFutureStandardLibraryUrllib, self).transform(node, results)
        touch_import_top(u'future', u'standard_library', root)
        return result
