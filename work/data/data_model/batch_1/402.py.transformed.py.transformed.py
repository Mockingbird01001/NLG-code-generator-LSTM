
from os.path import dirname, join, exists, sep
from lib2to3.fixes.fix_import import FixImport
from lib2to3.fixer_util import FromImport, syms
from lib2to3.fixes.fix_import import traverse_imports
from libfuturize.fixer_util import future_import
class FixAbsoluteImport(FixImport):
    run_order = 9
    def transform(self, node, results):
        if self.skip:
            return
        imp = results['imp']
        if node.type == syms.import_from:
            while not hasattr(imp, 'value'):
                imp = imp.children[0]
            if self.probably_a_local_import(imp.value):
                imp.value = u"." + imp.value
                imp.changed()
                future_import(u"absolute_import", node)
        else:
            have_local = False
            have_absolute = False
            for mod_name in traverse_imports(imp):
                if self.probably_a_local_import(mod_name):
                    have_local = True
                else:
                    have_absolute = True
            if have_absolute:
                if have_local:
                    self.warning(node, "absolute and local imports together")
                return
            new = FromImport(u".", [imp])
            new.prefix = node.prefix
            future_import(u"absolute_import", node)
            return new
    def probably_a_local_import(self, imp_name):
        if imp_name.startswith(u"."):
            return False
        imp_name = imp_name.split(u".", 1)[0]
        base_path = dirname(self.filename)
        base_path = join(base_path, imp_name)
        if not exists(join(dirname(base_path), "__init__.py")):
            return False
        for ext in [".py", sep, ".pyc", ".so", ".sl", ".pyd", ".pyx"]:
            if exists(base_path + ext):
                return True
        return False
