
from lib2to3.fixes.fix_xrange import FixXrange
from libfuturize.fixer_util import touch_import_top
class FixXrangeWithImport(FixXrange):
    def transform(self, node, results):
        result = super(FixXrangeWithImport, self).transform(node, results)
        touch_import_top('builtins', 'range', node)
        return result
