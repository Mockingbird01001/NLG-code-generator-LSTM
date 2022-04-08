
from lib2to3 import fixer_base
from lib2to3.fixer_util import Name
class FixGetcwd(fixer_base.BaseFix):
    PATTERN =
    def transform(self, node, results):
        if u"name" in results:
            name = results[u"name"]
            name.replace(Name(u"getcwdu", prefix=name.prefix))
        elif u"bad" in results:
            self.cannot_convert(node, u"import os, use os.getcwd() instead.")
            return
        else:
            raise ValueError(u"For some reason, the pattern matcher failed.")
