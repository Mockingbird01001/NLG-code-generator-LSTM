
"""Tests for ast_edits which is used in tf upgraders.
All of the tests assume that we want to change from an API containing
    import foo as f
    def f(a, b, kw1, kw2): ...
    def g(a, b, kw1, c, kw1_alias): ...
    def g2(a, b, kw1, c, d, kw1_alias): ...
    def h(a, kw1, kw2, kw1_alias, kw2_alias): ...
and the changes to the API consist of renaming, reordering, and/or removing
arguments. Thus, we want to be able to generate changes to produce each of the
following new APIs:
    import bar as f
    def f(a, b, kw1, kw3): ...
    def f(a, b, kw2, kw1): ...
    def f(a, b, kw3, kw1): ...
    def g(a, b, kw1, c): ...
    def g(a, b, c, kw1): ...
    def g2(a, b, kw1, c, d): ...
    def g2(a, b, c, d, kw1): ...
    def h(a, kw1, kw2): ...
"""
import ast
import os
import six
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test as test_lib
from tensorflow.tools.compatibility import ast_edits
class ModuleDeprecationSpec(ast_edits.NoUpdateSpec):
  def __init__(self):
    ast_edits.NoUpdateSpec.__init__(self)
    self.module_deprecations.update({"a.b": (ast_edits.ERROR, "a.b is evil.")})
class RenameKeywordSpec(ast_edits.NoUpdateSpec):
  """A specification where kw2 gets renamed to kw3.
  The new API is
    def f(a, b, kw1, kw3): ...
  """
  def __init__(self):
    ast_edits.NoUpdateSpec.__init__(self)
    self.update_renames()
  def update_renames(self):
    self.function_keyword_renames["f"] = {"kw2": "kw3"}
class ReorderKeywordSpec(ast_edits.NoUpdateSpec):
  """A specification where kw2 gets moved in front of kw1.
  The new API is
    def f(a, b, kw2, kw1): ...
  """
  def __init__(self):
    ast_edits.NoUpdateSpec.__init__(self)
    self.update_reorders()
  def update_reorders(self):
    self.function_reorders["f"] = ["a", "b", "kw1", "kw2"]
class ReorderAndRenameKeywordSpec(ReorderKeywordSpec, RenameKeywordSpec):
  """A specification where kw2 gets moved in front of kw1 and is changed to kw3.
  The new API is
    def f(a, b, kw3, kw1): ...
  """
  def __init__(self):
    ReorderKeywordSpec.__init__(self)
    RenameKeywordSpec.__init__(self)
    self.update_renames()
    self.update_reorders()
class RemoveDeprecatedAliasKeyword(ast_edits.NoUpdateSpec):
  """A specification where kw1_alias is removed in g.
  The new API is
    def g(a, b, kw1, c): ...
    def g2(a, b, kw1, c, d): ...
  """
  def __init__(self):
    ast_edits.NoUpdateSpec.__init__(self)
    self.function_keyword_renames["g"] = {"kw1_alias": "kw1"}
    self.function_keyword_renames["g2"] = {"kw1_alias": "kw1"}
class RemoveDeprecatedAliasAndReorderRest(RemoveDeprecatedAliasKeyword):
  """A specification where kw1_alias is removed in g.
  The new API is
    def g(a, b, c, kw1): ...
    def g2(a, b, c, d, kw1): ...
  """
  def __init__(self):
    RemoveDeprecatedAliasKeyword.__init__(self)
    self.function_reorders["g"] = ["a", "b", "kw1", "c"]
    self.function_reorders["g2"] = ["a", "b", "kw1", "c", "d"]
class RemoveMultipleKeywordArguments(ast_edits.NoUpdateSpec):
  """A specification where both keyword aliases are removed from h.
  The new API is
    def h(a, kw1, kw2): ...
  """
  def __init__(self):
    ast_edits.NoUpdateSpec.__init__(self)
    self.function_keyword_renames["h"] = {
        "kw1_alias": "kw1",
        "kw2_alias": "kw2",
    }
class RenameImports(ast_edits.NoUpdateSpec):
  def __init__(self):
    ast_edits.NoUpdateSpec.__init__(self)
    self.import_renames = {
        "foo": ast_edits.ImportRename(
            "bar",
            excluded_prefixes=["foo.baz"])
    }
class TestAstEdits(test_util.TensorFlowTestCase):
  def _upgrade(self, spec, old_file_text):
    in_file = six.StringIO(old_file_text)
    out_file = six.StringIO()
    upgrader = ast_edits.ASTCodeUpgrader(spec)
    count, report, errors = (
        upgrader.process_opened_file("test.py", in_file,
                                     "test_out.py", out_file))
    return (count, report, errors), out_file.getvalue()
  def testModuleDeprecation(self):
    text = "a.b.c(a.b.x)"
    (_, _, errors), new_text = self._upgrade(ModuleDeprecationSpec(), text)
    self.assertEqual(text, new_text)
    self.assertIn("Using member a.b.c", errors[0])
    self.assertIn("1:0", errors[0])
    self.assertIn("Using member a.b.c", errors[0])
    self.assertIn("1:6", errors[1])
  def testNoTransformIfNothingIsSupplied(self):
    text = "f(a, b, kw1=c, kw2=d)\n"
    _, new_text = self._upgrade(ast_edits.NoUpdateSpec(), text)
    self.assertEqual(new_text, text)
    text = "f(a, b, c, d)\n"
    _, new_text = self._upgrade(ast_edits.NoUpdateSpec(), text)
    self.assertEqual(new_text, text)
  def testKeywordRename(self):
    text = "f(a, b, kw1=c, kw2=d)\n"
    expected = "f(a, b, kw1=c, kw3=d)\n"
    (_, report, _), new_text = self._upgrade(RenameKeywordSpec(), text)
    self.assertEqual(new_text, expected)
    self.assertNotIn("Manual check required", report)
    text = "f(a, b, c, d)\n"
    (_, report, _), new_text = self._upgrade(RenameKeywordSpec(), text)
    self.assertEqual(new_text, text)
    self.assertNotIn("Manual check required", report)
    text = "f(a, *args)\n"
    (_, report, _), _ = self._upgrade(RenameKeywordSpec(), text)
    self.assertNotIn("Manual check required", report)
    text = "f(a, b, kw1=c, **kwargs)\n"
    (_, report, _), _ = self._upgrade(RenameKeywordSpec(), text)
    self.assertIn("Manual check required", report)
  def testKeywordReorderWithParens(self):
    text = "f((a), ( ( b ) ))\n"
    acceptable_outputs = [
        text,
        "f(a=(a), b=( ( b ) ))\n",
        "f(a=(a), b=((b)))\n",
    ]
    _, new_text = self._upgrade(ReorderKeywordSpec(), text)
    self.assertIn(new_text, acceptable_outputs)
  def testKeywordReorder(self):
    text = "f(a, b, kw1=c, kw2=d)\n"
    acceptable_outputs = [
        text,
        "f(a, b, kw2=d, kw1=c)\n",
        "f(a=a, b=b, kw1=c, kw2=d)\n",
        "f(a=a, b=b, kw2=d, kw1=c)\n",
    ]
    (_, report, _), new_text = self._upgrade(ReorderKeywordSpec(), text)
    self.assertIn(new_text, acceptable_outputs)
    self.assertNotIn("Manual check required", report)
    text = "f(a, b, c, d)\n"
    acceptable_outputs = [
        "f(a, b, d, c)\n",
        "f(a=a, b=b, kw1=c, kw2=d)\n",
        "f(a=a, b=b, kw2=d, kw1=c)\n",
    ]
    (_, report, _), new_text = self._upgrade(ReorderKeywordSpec(), text)
    self.assertIn(new_text, acceptable_outputs)
    self.assertNotIn("Manual check required", report)
    text = "f(a, b, *args)\n"
    (_, report, _), _ = self._upgrade(ReorderKeywordSpec(), text)
    self.assertIn("Manual check required", report)
    text = "f(a, b, kw1=c, **kwargs)\n"
    (_, report, _), _ = self._upgrade(ReorderKeywordSpec(), text)
    self.assertNotIn("Manual check required", report)
  def testKeywordReorderAndRename(self):
    text = "f(a, b, kw1=c, kw2=d)\n"
    acceptable_outputs = [
        "f(a, b, kw3=d, kw1=c)\n",
        "f(a=a, b=b, kw1=c, kw3=d)\n",
        "f(a=a, b=b, kw3=d, kw1=c)\n",
    ]
    (_, report, _), new_text = self._upgrade(
        ReorderAndRenameKeywordSpec(), text)
    self.assertIn(new_text, acceptable_outputs)
    self.assertNotIn("Manual check required", report)
    text = "f(a, b, c, d)\n"
    acceptable_outputs = [
        "f(a, b, d, c)\n",
        "f(a=a, b=b, kw1=c, kw3=d)\n",
        "f(a=a, b=b, kw3=d, kw1=c)\n",
    ]
    (_, report, _), new_text = self._upgrade(
        ReorderAndRenameKeywordSpec(), text)
    self.assertIn(new_text, acceptable_outputs)
    self.assertNotIn("Manual check required", report)
    text = "f(a, *args, kw1=c)\n"
    (_, report, _), _ = self._upgrade(ReorderAndRenameKeywordSpec(), text)
    self.assertIn("Manual check required", report)
    text = "f(a, b, kw1=c, **kwargs)\n"
    (_, report, _), _ = self._upgrade(ReorderAndRenameKeywordSpec(), text)
    self.assertIn("Manual check required", report)
  def testRemoveDeprecatedKeywordAlias(self):
    text = "g(a, b, kw1=x, c=c)\n"
    acceptable_outputs = [
        text,
        "g(a=a, b=b, kw1=x, c=c)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasKeyword(), text)
    self.assertIn(new_text, acceptable_outputs)
    text = "g(a, b, x, c)\n"
    _, new_text = self._upgrade(RemoveDeprecatedAliasKeyword(), text)
    self.assertEqual(new_text, text)
    text = "g(a, b, kw1_alias=x, c=c)\n"
    acceptable_outputs = [
        "g(a, b, kw1=x, c=c)\n",
        "g(a, b, c=c, kw1=x)\n",
        "g(a=a, b=b, kw1=x, c=c)\n",
        "g(a=a, b=b, c=c, kw1=x)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasKeyword(), text)
    self.assertIn(new_text, acceptable_outputs)
    text = "g(a, b, c=c, kw1_alias=x)\n"
    acceptable_outputs = [
        "g(a, b, kw1=x, c=c)\n",
        "g(a, b, c=c, kw1=x)\n",
        "g(a=a, b=b, kw1=x, c=c)\n",
        "g(a=a, b=b, c=c, kw1=x)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasKeyword(), text)
    self.assertIn(new_text, acceptable_outputs)
  def testRemoveDeprecatedKeywordAndReorder(self):
    text = "g(a, b, kw1=x, c=c)\n"
    acceptable_outputs = [
        "g(a, b, c=c, kw1=x)\n",
        "g(a=a, b=b, kw1=x, c=c)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasAndReorderRest(), text)
    self.assertIn(new_text, acceptable_outputs)
    text = "g(a, b, x, c)\n"
    acceptable_outputs = [
        "g(a, b, c, x)\n",
        "g(a=a, b=b, kw1=x, c=c)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasAndReorderRest(), text)
    self.assertIn(new_text, acceptable_outputs)
    text = "g(a, b, kw1_alias=x, c=c)\n"
    acceptable_outputs = [
        "g(a, b, kw1=x, c=c)\n",
        "g(a, b, c=c, kw1=x)\n",
        "g(a=a, b=b, kw1=x, c=c)\n",
        "g(a=a, b=b, c=c, kw1=x)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasKeyword(), text)
    self.assertIn(new_text, acceptable_outputs)
    text = "g(a, b, c=c, kw1_alias=x)\n"
    acceptable_outputs = [
        "g(a, b, kw1=x, c=c)\n",
        "g(a, b, c=c, kw1=x)\n",
        "g(a=a, b=b, kw1=x, c=c)\n",
        "g(a=a, b=b, c=c, kw1=x)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasKeyword(), text)
    self.assertIn(new_text, acceptable_outputs)
  def testRemoveDeprecatedKeywordAndReorder2(self):
    text = "g2(a, b, kw1=x, c=c, d=d)\n"
    acceptable_outputs = [
        "g2(a, b, c=c, d=d, kw1=x)\n",
        "g2(a=a, b=b, kw1=x, c=c, d=d)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasAndReorderRest(), text)
    self.assertIn(new_text, acceptable_outputs)
    text = "g2(a, b, x, c, d)\n"
    acceptable_outputs = [
        "g2(a, b, c, d, x)\n",
        "g2(a=a, b=b, kw1=x, c=c, d=d)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasAndReorderRest(), text)
    self.assertIn(new_text, acceptable_outputs)
    text = "g2(a, b, kw1_alias=x, c=c, d=d)\n"
    acceptable_outputs = [
        "g2(a, b, kw1=x, c=c, d=d)\n",
        "g2(a, b, c=c, d=d, kw1=x)\n",
        "g2(a=a, b=b, kw1=x, c=c, d=d)\n",
        "g2(a=a, b=b, c=c, d=d, kw1=x)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasKeyword(), text)
    self.assertIn(new_text, acceptable_outputs)
    text = "g2(a, b, d=d, c=c, kw1_alias=x)\n"
    acceptable_outputs = [
        "g2(a, b, kw1=x, c=c, d=d)\n",
        "g2(a, b, c=c, d=d, kw1=x)\n",
        "g2(a, b, d=d, c=c, kw1=x)\n",
        "g2(a=a, b=b, kw1=x, c=c, d=d)\n",
        "g2(a=a, b=b, c=c, d=d, kw1=x)\n",
        "g2(a=a, b=b, d=d, c=c, kw1=x)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasKeyword(), text)
    self.assertIn(new_text, acceptable_outputs)
  def testRemoveMultipleKeywords(self):
    text = "h(a, kw1=x, kw2=y)\n"
    _, new_text = self._upgrade(RemoveMultipleKeywordArguments(), text)
    self.assertEqual(new_text, text)
    text = "h(a, x, y)\n"
    _, new_text = self._upgrade(RemoveMultipleKeywordArguments(), text)
    self.assertEqual(new_text, text)
    text = "h(a, kw1_alias=x, kw2_alias=y)\n"
    acceptable_outputs = [
        "h(a, x, y)\n",
        "h(a, kw1=x, kw2=y)\n",
        "h(a=a, kw1=x, kw2=y)\n",
        "h(a, kw2=y, kw1=x)\n",
        "h(a=a, kw2=y, kw1=x)\n",
    ]
    _, new_text = self._upgrade(RemoveMultipleKeywordArguments(), text)
    self.assertIn(new_text, acceptable_outputs)
    text = "h(a, kw2_alias=y, kw1_alias=x)\n"
    _, new_text = self._upgrade(RemoveMultipleKeywordArguments(), text)
    self.assertIn(new_text, acceptable_outputs)
    text = "h(a, kw1=x, kw2_alias=y)\n"
    _, new_text = self._upgrade(RemoveMultipleKeywordArguments(), text)
    self.assertIn(new_text, acceptable_outputs)
  def testUnrestrictedFunctionWarnings(self):
    class FooWarningSpec(ast_edits.NoUpdateSpec):
      def __init__(self):
        ast_edits.NoUpdateSpec.__init__(self)
        self.function_warnings = {"*.foo": (ast_edits.WARNING, "not good")}
    texts = ["object.foo()", "get_object().foo()",
             "get_object().foo()", "object.foo().bar()"]
    for text in texts:
      (_, report, _), _ = self._upgrade(FooWarningSpec(), text)
      self.assertIn("not good", report)
    false_alarms = ["foo", "foo()", "foo.bar()", "obj.run_foo()", "obj.foo"]
    for text in false_alarms:
      (_, report, _), _ = self._upgrade(FooWarningSpec(), text)
      self.assertNotIn("not good", report)
  def testFullNameNode(self):
    t = ast_edits.full_name_node("a.b.c")
    self.assertEqual(
        ast.dump(t),
        "Attribute(value=Attribute(value=Name(id='a', ctx=Load()), attr='b', "
        "ctx=Load()), attr='c', ctx=Load())")
  def testImport(self):
    text = "import foo as f"
    expected_text = "import bar as f"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)
    text = "import foo"
    expected_text = "import bar as foo"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)
    text = "import foo.test"
    expected_text = "import bar.test"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)
    text = "import foo.test as t"
    expected_text = "import bar.test as t"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)
    text = "import foo as f, a as b"
    expected_text = "import bar as f, a as b"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)
  def testFromImport(self):
    text = "from foo import a"
    expected_text = "from bar import a"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)
    text = "from foo.a import b"
    expected_text = "from bar.a import b"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)
    text = "from foo import *"
    expected_text = "from bar import *"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)
    text = "from foo import a, b"
    expected_text = "from bar import a, b"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)
  def testImport_NoChangeNeeded(self):
    text = "import bar as b"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(text, new_text)
  def testFromImport_NoChangeNeeded(self):
    text = "from bar import a as b"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(text, new_text)
  def testExcludedImport(self):
    text = "import foo.baz"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(text, new_text)
    text = "import foo.baz as a"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(text, new_text)
    text = "from foo import baz as a"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(text, new_text)
    text = "from foo.baz import a"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(text, new_text)
  def testMultipleImports(self):
    text = "import foo.bar as a, foo.baz as b, foo.baz.c, foo.d"
    expected_text = "import bar.bar as a, foo.baz as b, foo.baz.c, bar.d"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)
    text = "from foo import baz, a, c"
    expected_text =
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)
  def testImportInsideFunction(self):
    text = """
def t():
  from c import d
  from foo import baz, a
  from e import y
def t():
  from c import d
  from foo import baz
  from bar import a
  from e import y
"""
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)
  def testUpgradeInplaceWithSymlink(self):
    if os.name == "nt":
      self.skipTest("os.symlink doesn't work uniformly on Windows.")
    upgrade_dir = os.path.join(self.get_temp_dir(), "foo")
    os.mkdir(upgrade_dir)
    file_a = os.path.join(upgrade_dir, "a.py")
    file_b = os.path.join(upgrade_dir, "b.py")
    with open(file_a, "a") as f:
      f.write("import foo as f")
    os.symlink(file_a, file_b)
    upgrader = ast_edits.ASTCodeUpgrader(RenameImports())
    upgrader.process_tree_inplace(upgrade_dir)
    self.assertTrue(os.path.islink(file_b))
    self.assertEqual(file_a, os.readlink(file_b))
    with open(file_a, "r") as f:
      self.assertEqual("import bar as f", f.read())
  def testUpgradeInPlaceWithSymlinkInDifferentDir(self):
    if os.name == "nt":
      self.skipTest("os.symlink doesn't work uniformly on Windows.")
    upgrade_dir = os.path.join(self.get_temp_dir(), "foo")
    other_dir = os.path.join(self.get_temp_dir(), "bar")
    os.mkdir(upgrade_dir)
    os.mkdir(other_dir)
    file_c = os.path.join(other_dir, "c.py")
    file_d = os.path.join(upgrade_dir, "d.py")
    with open(file_c, "a") as f:
      f.write("import foo as f")
    os.symlink(file_c, file_d)
    upgrader = ast_edits.ASTCodeUpgrader(RenameImports())
    upgrader.process_tree_inplace(upgrade_dir)
    self.assertTrue(os.path.islink(file_d))
    self.assertEqual(file_c, os.readlink(file_d))
    with open(file_c, "r") as f:
      self.assertEqual("import foo as f", f.read())
  def testUpgradeCopyWithSymlink(self):
    if os.name == "nt":
      self.skipTest("os.symlink doesn't work uniformly on Windows.")
    upgrade_dir = os.path.join(self.get_temp_dir(), "foo")
    output_dir = os.path.join(self.get_temp_dir(), "bar")
    os.mkdir(upgrade_dir)
    file_a = os.path.join(upgrade_dir, "a.py")
    file_b = os.path.join(upgrade_dir, "b.py")
    with open(file_a, "a") as f:
      f.write("import foo as f")
    os.symlink(file_a, file_b)
    upgrader = ast_edits.ASTCodeUpgrader(RenameImports())
    upgrader.process_tree(upgrade_dir, output_dir, copy_other_files=True)
    new_file_a = os.path.join(output_dir, "a.py")
    new_file_b = os.path.join(output_dir, "b.py")
    self.assertTrue(os.path.islink(new_file_b))
    self.assertEqual(new_file_a, os.readlink(new_file_b))
    with open(new_file_a, "r") as f:
      self.assertEqual("import bar as f", f.read())
  def testUpgradeCopyWithSymlinkInDifferentDir(self):
    if os.name == "nt":
      self.skipTest("os.symlink doesn't work uniformly on Windows.")
    upgrade_dir = os.path.join(self.get_temp_dir(), "foo")
    other_dir = os.path.join(self.get_temp_dir(), "bar")
    output_dir = os.path.join(self.get_temp_dir(), "baz")
    os.mkdir(upgrade_dir)
    os.mkdir(other_dir)
    file_a = os.path.join(other_dir, "a.py")
    file_b = os.path.join(upgrade_dir, "b.py")
    with open(file_a, "a") as f:
      f.write("import foo as f")
    os.symlink(file_a, file_b)
    upgrader = ast_edits.ASTCodeUpgrader(RenameImports())
    upgrader.process_tree(upgrade_dir, output_dir, copy_other_files=True)
    new_file_b = os.path.join(output_dir, "b.py")
    self.assertTrue(os.path.islink(new_file_b))
    self.assertEqual(file_a, os.readlink(new_file_b))
    with open(file_a, "r") as f:
      self.assertEqual("import foo as f", f.read())
if __name__ == "__main__":
  test_lib.main()
