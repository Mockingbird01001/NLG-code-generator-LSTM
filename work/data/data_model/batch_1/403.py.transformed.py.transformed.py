
from __future__ import unicode_literals
from lib2to3 import fixer_base
from libfuturize.fixer_util import touch_import_top
class FixAddAllFutureBuiltins(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = "file_input"
    run_order = 1
    def transform(self, node, results):
        touch_import_top(u'builtins', '*', node)
