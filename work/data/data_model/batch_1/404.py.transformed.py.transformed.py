
from lib2to3 import fixer_base
from libfuturize.fixer_util import future_import
class FixAddAllFutureImports(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = "file_input"
    run_order = 1
    def transform(self, node, results):
        future_import(u"absolute_import", node)
        future_import(u"division", node)
        future_import(u"print_function", node)
        future_import(u"unicode_literals", node)
