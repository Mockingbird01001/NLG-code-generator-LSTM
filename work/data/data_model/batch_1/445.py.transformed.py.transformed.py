
from lib2to3 import fixer_base
from lib2to3.fixer_util import Name, attr_chain
from lib2to3.fixes.fix_imports import alternates, build_pattern, FixImports
MAPPING = {'UserDict':  'collections',
}
class FixUserdict(FixImports):
    BM_compatible = True
    keep_line_order = True
    mapping = MAPPING
    run_order = 6
    def build_pattern(self):
        return "|".join(build_pattern(self.mapping))
    def compile_pattern(self):
        self.PATTERN = self.build_pattern()
        super(FixImports, self).compile_pattern()
    def match(self, node):
        match = super(FixImports, self).match
        results = match(node)
        if results:
            if "bare_with_attr" not in results and                    any(match(obj) for obj in attr_chain(node, "parent")):
                return False
            return results
        return False
    def start_tree(self, tree, filename):
        super(FixImports, self).start_tree(tree, filename)
        self.replace = {}
    def transform(self, node, results):
        import_mod = results.get("module_name")
        if import_mod:
            mod_name = import_mod.value
            new_name = unicode(self.mapping[mod_name])
            import_mod.replace(Name(new_name, prefix=import_mod.prefix))
            if "name_import" in results:
                self.replace[mod_name] = new_name
            if "multiple_imports" in results:
                results = self.match(node)
                if results:
                    self.transform(node, results)
        else:
            bare_name = results["bare_with_attr"][0]
            new_name = self.replace.get(bare_name.value)
            if new_name:
                bare_name.replace(Name(new_name, prefix=bare_name.prefix))
