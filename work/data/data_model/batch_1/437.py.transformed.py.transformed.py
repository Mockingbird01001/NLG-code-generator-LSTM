
from libfuturize.fixes.fix_print import FixPrint
from libfuturize.fixer_util import future_import
class FixPrintWithImport(FixPrint):
    run_order = 7
    def transform(self, node, results):
        future_import(u'print_function', node)
        n_stmt = super(FixPrintWithImport, self).transform(node, results)
        return n_stmt
