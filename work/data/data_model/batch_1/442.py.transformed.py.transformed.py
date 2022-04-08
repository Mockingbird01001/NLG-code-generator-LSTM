
from lib2to3.pgen2 import token
from lib2to3 import fixer_base
_mapping = {u"unichr" : u"chr", u"unicode" : u"str"}
class FixUnicodeKeepU(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = "'unicode' | 'unichr'"
    def transform(self, node, results):
        if node.type == token.NAME:
            new = node.clone()
            new.value = _mapping[node.value]
            return new
