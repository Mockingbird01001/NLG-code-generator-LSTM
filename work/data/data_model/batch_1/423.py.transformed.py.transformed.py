
from lib2to3 import fixer_base
from lib2to3.fixer_util import Name, is_probably_builtin, Newline, does_tree_import
from lib2to3.pygram import python_symbols as syms
from lib2to3.pgen2 import token
from lib2to3.pytree import Node, Leaf
from libfuturize.fixer_util import touch_import_top
MAPPING = {u"reprlib": u"repr",
           u"winreg": u"_winreg",
           u"configparser": u"ConfigParser",
           u"copyreg": u"copy_reg",
           u"queue": u"Queue",
           u"socketserver": u"SocketServer",
           u"_markupbase": u"markupbase",
           u"test.support": u"test.test_support",
           u"dbm.bsd": u"dbhash",
           u"dbm.ndbm": u"dbm",
           u"dbm.dumb": u"dumbdbm",
           u"dbm.gnu": u"gdbm",
           u"html.parser": u"HTMLParser",
           u"html.entities": u"htmlentitydefs",
           u"http.client": u"httplib",
           u"http.cookies": u"Cookie",
           u"http.cookiejar": u"cookielib",
           u"tkinter.dialog": u"Dialog",
           u"tkinter._fix": u"FixTk",
           u"tkinter.scrolledtext": u"ScrolledText",
           u"tkinter.tix": u"Tix",
           u"tkinter.constants": u"Tkconstants",
           u"tkinter.dnd": u"Tkdnd",
           u"tkinter.__init__": u"Tkinter",
           u"tkinter.colorchooser": u"tkColorChooser",
           u"tkinter.commondialog": u"tkCommonDialog",
           u"tkinter.font": u"tkFont",
           u"tkinter.ttk": u"ttk",
           u"tkinter.messagebox": u"tkMessageBox",
           u"tkinter.turtle": u"turtle",
           u"urllib.robotparser": u"robotparser",
           u"xmlrpc.client": u"xmlrpclib",
           u"builtins": u"__builtin__",
}
simple_name_match = u"name='%s'"
subname_match = u"attr='%s'"
dotted_name_match = u"dotted_name=dotted_name< %s '.' %s >"
power_onename_match = u"%s"
power_twoname_match = u"power< %s trailer< '.' %s > any* >"
power_subname_match = u"power< %s any* >"
from_import_match = u"from_import=import_from< 'from' %s 'import' imported=any >"
from_import_submod_match = u"from_import_submod=import_from< 'from' %s 'import' (%s | import_as_name< %s 'as' renamed=any > | import_as_names< any* (%s | import_as_name< %s 'as' renamed=any >) any* > ) >"
name_import_match = u"name_import=import_name< 'import' %s > | name_import=import_name< 'import' dotted_as_name< %s 'as' renamed=any > >"
multiple_name_import_match = u"name_import=import_name< 'import' dotted_as_names< names=any* > >"
def all_patterns(name):
    if u'.' in name:
        name, attr = name.split(u'.', 1)
        simple_name = simple_name_match % (name)
        simple_attr = subname_match % (attr)
        dotted_name = dotted_name_match % (simple_name, simple_attr)
        i_from = from_import_match % (dotted_name)
        i_from_submod = from_import_submod_match % (simple_name, simple_attr, simple_attr, simple_attr, simple_attr)
        i_name = name_import_match % (dotted_name, dotted_name)
        u_name = power_twoname_match % (simple_name, simple_attr)
        u_subname = power_subname_match % (simple_attr)
        return u' | \n'.join((i_name, i_from, i_from_submod, u_name, u_subname))
    else:
        simple_name = simple_name_match % (name)
        i_name = name_import_match % (simple_name, simple_name)
        i_from = from_import_match % (simple_name)
        u_name = power_onename_match % (simple_name)
        return u' | \n'.join((i_name, i_from, u_name))
class FixImports(fixer_base.BaseFix):
    PATTERN = u' | \n'.join([all_patterns(name) for name in MAPPING])
    PATTERN = u' | \n'.join((PATTERN, multiple_name_import_match))
    def transform(self, node, results):
        touch_import_top(u'future', u'standard_library', node)
