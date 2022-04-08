
from __future__ import unicode_literals
from lib2to3 import fixer_base
from lib2to3.pygram import python_symbols as syms
from lib2to3.fixer_util import Name, Call, in_special_context
from libfuturize.fixer_util import touch_import_top
replaced_builtins = '''filter map zip
                       ascii chr hex input next oct open round super
                       bytes dict int range str'''.split()
expression = '|'.join(["name='{0}'".format(name) for name in replaced_builtins])
class FixFutureBuiltins(fixer_base.BaseFix):
    BM_compatible = True
    run_order = 9
    PATTERN = """
              power<
                 ({0}) trailer< '(' args=[any] ')' >
              rest=any* >
              """.format(expression)
    def transform(self, node, results):
        name = results["name"]
        touch_import_top(u'builtins', name.value, node)
