
from lib2to3 import fixer_base
from lib2to3.fixer_util import Name
class FixMemoryview(fixer_base.BaseFix):
    explicit = True
    PATTERN = u"""
              power< name='memoryview' trailer< '(' [any] ')' >
              rest=any* >
              """
    def transform(self, node, results):
        name = results[u"name"]
        name.replace(Name(u"buffer", prefix=name.prefix))
