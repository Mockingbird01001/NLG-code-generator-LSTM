
from lib2to3 import fixer_base
from lib2to3.fixer_util import Name, String, FromImport, Newline, Comma
from libfuturize.fixer_util import touch_import_top
TK_BASE_NAMES = (u'ACTIVE', u'ALL', u'ANCHOR', u'ARC',u'BASELINE', u'BEVEL', u'BOTH',
                 u'BOTTOM', u'BROWSE', u'BUTT', u'CASCADE', u'CENTER', u'CHAR',
                 u'CHECKBUTTON', u'CHORD', u'COMMAND', u'CURRENT', u'DISABLED',
                 u'DOTBOX', u'E', u'END', u'EW', u'EXCEPTION', u'EXTENDED', u'FALSE',
                 u'FIRST', u'FLAT', u'GROOVE', u'HIDDEN', u'HORIZONTAL', u'INSERT',
                 u'INSIDE', u'LAST', u'LEFT', u'MITER', u'MOVETO', u'MULTIPLE', u'N',
                 u'NE', u'NO', u'NONE', u'NORMAL', u'NS', u'NSEW', u'NUMERIC', u'NW',
                 u'OFF', u'ON', u'OUTSIDE', u'PAGES', u'PIESLICE', u'PROJECTING',
                 u'RADIOBUTTON', u'RAISED', u'READABLE', u'RIDGE', u'RIGHT',
                 u'ROUND', u'S', u'SCROLL', u'SE', u'SEL', u'SEL_FIRST', u'SEL_LAST',
                 u'SEPARATOR', u'SINGLE', u'SOLID', u'SUNKEN', u'SW', u'StringTypes',
                 u'TOP', u'TRUE', u'TclVersion', u'TkVersion', u'UNDERLINE',
                 u'UNITS', u'VERTICAL', u'W', u'WORD', u'WRITABLE', u'X', u'Y', u'YES',
                 u'wantobjects')
PY2MODULES = {
              u'urllib2' : (
                  u'AbstractBasicAuthHandler', u'AbstractDigestAuthHandler',
                  u'AbstractHTTPHandler', u'BaseHandler', u'CacheFTPHandler',
                  u'FTPHandler', u'FileHandler', u'HTTPBasicAuthHandler',
                  u'HTTPCookieProcessor', u'HTTPDefaultErrorHandler',
                  u'HTTPDigestAuthHandler', u'HTTPError', u'HTTPErrorProcessor',
                  u'HTTPHandler', u'HTTPPasswordMgr',
                  u'HTTPPasswordMgrWithDefaultRealm', u'HTTPRedirectHandler',
                  u'HTTPSHandler', u'OpenerDirector', u'ProxyBasicAuthHandler',
                  u'ProxyDigestAuthHandler', u'ProxyHandler', u'Request',
                  u'StringIO', u'URLError', u'UnknownHandler', u'addinfourl',
                  u'build_opener', u'install_opener', u'parse_http_list',
                  u'parse_keqv_list', u'randombytes', u'request_host', u'urlopen'),
              u'urllib' : (
                  u'ContentTooShortError', u'FancyURLopener',u'URLopener',
                  u'basejoin', u'ftperrors', u'getproxies',
                  u'getproxies_environment', u'localhost', u'pathname2url',
                  u'quote', u'quote_plus', u'splitattr', u'splithost',
                  u'splitnport', u'splitpasswd', u'splitport', u'splitquery',
                  u'splittag', u'splittype', u'splituser', u'splitvalue',
                  u'thishost', u'unquote', u'unquote_plus', u'unwrap',
                  u'url2pathname', u'urlcleanup', u'urlencode', u'urlopen',
                  u'urlretrieve',),
              u'urlparse' : (
                  u'parse_qs', u'parse_qsl', u'urldefrag', u'urljoin',
                  u'urlparse', u'urlsplit', u'urlunparse', u'urlunsplit'),
              u'dbm' : (
                  u'ndbm', u'gnu', u'dumb'),
              u'anydbm' : (
                  u'error', u'open'),
              u'whichdb' : (
                  u'whichdb',),
              u'BaseHTTPServer' : (
                  u'BaseHTTPRequestHandler', u'HTTPServer'),
              u'CGIHTTPServer' : (
                  u'CGIHTTPRequestHandler',),
              u'SimpleHTTPServer' : (
                  u'SimpleHTTPRequestHandler',),
              u'FileDialog' : TK_BASE_NAMES + (
                  u'FileDialog', u'LoadFileDialog', u'SaveFileDialog',
                  u'dialogstates', u'test'),
              u'tkFileDialog' : (
                  u'Directory', u'Open', u'SaveAs', u'_Dialog', u'askdirectory',
                  u'askopenfile', u'askopenfilename', u'askopenfilenames',
                  u'askopenfiles', u'asksaveasfile', u'asksaveasfilename'),
              u'SimpleDialog' : TK_BASE_NAMES + (
                  u'SimpleDialog',),
              u'tkSimpleDialog' : TK_BASE_NAMES + (
                  u'askfloat', u'askinteger', u'askstring', u'Dialog'),
              u'SimpleXMLRPCServer' : (
                  u'CGIXMLRPCRequestHandler', u'SimpleXMLRPCDispatcher',
                  u'SimpleXMLRPCRequestHandler', u'SimpleXMLRPCServer',
                  u'list_public_methods', u'remove_duplicates',
                  u'resolve_dotted_attribute'),
              u'DocXMLRPCServer' : (
                  u'DocCGIXMLRPCRequestHandler', u'DocXMLRPCRequestHandler',
                  u'DocXMLRPCServer', u'ServerHTMLDoc',u'XMLRPCDocGenerator'),
                }
MAPPING = { u'urllib.request' :
                (u'urllib2', u'urllib'),
            u'urllib.error' :
                (u'urllib2', u'urllib'),
            u'urllib.parse' :
                (u'urllib2', u'urllib', u'urlparse'),
            u'dbm.__init__' :
                (u'anydbm', u'whichdb'),
            u'http.server' :
                (u'CGIHTTPServer', u'SimpleHTTPServer', u'BaseHTTPServer'),
            u'tkinter.filedialog' :
                (u'tkFileDialog', u'FileDialog'),
            u'tkinter.simpledialog' :
                (u'tkSimpleDialog', u'SimpleDialog'),
            u'xmlrpc.server' :
                (u'DocXMLRPCServer', u'SimpleXMLRPCServer'),
            }
simple_name = u"name='%s'"
simple_attr = u"attr='%s'"
simple_using = u"using='%s'"
dotted_name = u"dotted_name=dotted_name< %s '.' %s >"
power_twoname = u"pow=power< %s trailer< '.' %s > trailer< '.' using=any > any* >"
power_onename = u"pow=power< %s trailer< '.' using=any > any* >"
from_import = u"from_import=import_from< 'from' %s 'import' (import_as_name< using=any 'as' renamed=any> | in_list=import_as_names< using=any* > | using='*' | using=NAME) >"
name_import = u"name_import=import_name< 'import' (%s | in_list=dotted_as_names< imp_list=any* >) >"
name_import_rename = u"name_import_rename=dotted_as_name< %s 'as' renamed=any >"
from_import_rename = u"from_import_rename=import_from< 'from' %s 'import' (%s | import_as_name< %s 'as' renamed=any > | in_list=import_as_names< any* (%s | import_as_name< %s 'as' renamed=any >) any* >) >"
def all_modules_subpattern():
    names_dot_attrs = [mod.split(u".") for mod in MAPPING]
    ret = u"( " + u" | ".join([dotted_name % (simple_name % (mod[0]),
                                            simple_attr % (mod[1])) for mod in names_dot_attrs])
    ret += u" | "
    ret += u" | ".join([simple_name % (mod[0]) for mod in names_dot_attrs if mod[1] == u"__init__"]) + u" )"
    return ret
def build_import_pattern(mapping1, mapping2):
    yield from_import % (all_modules_subpattern())
    for py3k, py2k in mapping1.items():
        name, attr = py3k.split(u'.')
        s_name = simple_name % (name)
        s_attr = simple_attr % (attr)
        d_name = dotted_name % (s_name, s_attr)
        yield name_import % (d_name)
        yield power_twoname % (s_name, s_attr)
        if attr == u'__init__':
            yield name_import % (s_name)
            yield power_onename % (s_name)
        yield name_import_rename % (d_name)
        yield from_import_rename % (s_name, s_attr, s_attr, s_attr, s_attr)
class FixImports2(fixer_base.BaseFix):
    run_order = 4
    PATTERN = u" | \n".join(build_import_pattern(MAPPING, PY2MODULES))
    def transform(self, node, results):
        touch_import_top(u'future', u'standard_library', node)
