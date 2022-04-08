
from lib2to3 import fixer_base
from libfuturize.fixer_util import remove_future_import
class FixRemoveOldFutureImports(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = "file_input"
    run_order = 1
    def transform(self, node, results):
        remove_future_import(u"with_statement", node)
        remove_future_import(u"nested_scopes", node)
        remove_future_import(u"generators", node)
