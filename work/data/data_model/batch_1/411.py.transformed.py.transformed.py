
from lib2to3 import fixer_base
from libfuturize.fixer_util import token, future_import
def match_division(node):
    slash = token.SLASH
    return node.type == slash and not node.next_sibling.type == slash and                                  not node.prev_sibling.type == slash
class FixDivision(fixer_base.BaseFix):
    run_order = 4
    def match(self, node):
        return match_division(node)
    def transform(self, node, results):
        future_import(u"division", node)
