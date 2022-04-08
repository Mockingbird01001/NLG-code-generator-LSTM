
import lib2to3.fixes.fix_input
from lib2to3.fixer_util import does_tree_import
class FixInput(lib2to3.fixes.fix_input.FixInput):
    def transform(self, node, results):
        if does_tree_import('builtins', 'input', node):
            return
        return super(FixInput, self).transform(node, results)
