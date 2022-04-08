
from lib2to3.fixer_util import (FromImport, Newline, is_import,
                                find_root, does_tree_import, Comma)
from lib2to3.pytree import Leaf, Node
from lib2to3.pygram import python_symbols as syms, python_grammar
from lib2to3.pygram import token
from lib2to3.fixer_util import (Node, Call, Name, syms, Comma, Number)
import re
def canonical_fix_name(fix, avail_fixes):
    if ".fix_" in fix:
        return fix
    else:
        if fix.startswith('fix_'):
            fix = fix[4:]
        found = [f for f in avail_fixes
                 if f.endswith('fix_{0}'.format(fix))]
        if len(found) > 1:
            raise ValueError("Ambiguous fixer name. Choose a fully qualified "
                  "module name instead from these:\n" +
                  "\n".join("  " + myf for myf in found))
        elif len(found) == 0:
            raise ValueError("Unknown fixer. Use --list-fixes or -l for a list.")
        return found[0]
def Star(prefix=None):
    return Leaf(token.STAR, u'*', prefix=prefix)
def DoubleStar(prefix=None):
    return Leaf(token.DOUBLESTAR, u'**', prefix=prefix)
def Minus(prefix=None):
    return Leaf(token.MINUS, u'-', prefix=prefix)
def commatize(leafs):
    new_leafs = []
    for leaf in leafs:
        new_leafs.append(leaf)
        new_leafs.append(Comma())
    del new_leafs[-1]
    return new_leafs
def indentation(node):
    while node.parent is not None and node.parent.type != syms.suite:
        node = node.parent
    if node.parent is None:
        return u""
    if node.type == token.INDENT:
        return node.value
    elif node.prev_sibling is not None and node.prev_sibling.type == token.INDENT:
        return node.prev_sibling.value
    elif node.prev_sibling is None:
        return u""
    else:
        return node.prefix
def indentation_step(node):
    r = find_root(node)
    all_indents = set(i.value for i in r.pre_order() if i.type == token.INDENT)
    if not all_indents:
        return u"    "
    else:
        return min(all_indents)
def suitify(parent):
    for node in parent.children:
        if node.type == syms.suite:
            return
    for i, node in enumerate(parent.children):
        if node.type == token.COLON:
            break
    else:
        raise ValueError(u"No class suite and no ':'!")
    suite = Node(syms.suite, [Newline(), Leaf(token.INDENT, indentation(node) + indentation_step(node))])
    one_node = parent.children[i+1]
    one_node.remove()
    one_node.prefix = u''
    suite.append_child(one_node)
    parent.append_child(suite)
def NameImport(package, as_name=None, prefix=None):
    if prefix is None:
        prefix = u""
    children = [Name(u"import", prefix=prefix), package]
    if as_name is not None:
        children.extend([Name(u"as", prefix=u" "),
                         Name(as_name, prefix=u" ")])
    return Node(syms.import_name, children)
_compound_stmts = (syms.if_stmt, syms.while_stmt, syms.for_stmt, syms.try_stmt, syms.with_stmt)
_import_stmts = (syms.import_name, syms.import_from)
def import_binding_scope(node):
    assert node.type in _import_stmts
    test = node.next_sibling
    while test.type == token.SEMI:
        nxt = test.next_sibling
        if nxt.type == token.NEWLINE:
            break
        else:
            yield nxt
        test = nxt.next_sibling
    parent = node.parent
    assert parent.type == syms.simple_stmt
    test = parent.next_sibling
    while test is not None:
        yield test
        test = test.next_sibling
    context = parent.parent
    if context.type in _compound_stmts:
        c = context
        while c.next_sibling is not None:
            yield c.next_sibling
            c = c.next_sibling
        context = context.parent
    p = context.parent
    if p is None:
        return
    while p.type in _compound_stmts:
        if context.type == syms.suite:
            yield context
        context = context.next_sibling
        if context is None:
            context = p.parent
            p = context.parent
            if p is None:
                break
def ImportAsName(name, as_name, prefix=None):
    new_name = Name(name)
    new_as = Name(u"as", prefix=u" ")
    new_as_name = Name(as_name, prefix=u" ")
    new_node = Node(syms.import_as_name, [new_name, new_as, new_as_name])
    if prefix is not None:
        new_node.prefix = prefix
    return new_node
def is_docstring(node):
    return (node.type == syms.simple_stmt and
            len(node.children) > 0 and node.children[0].type == token.STRING)
def future_import(feature, node):
    root = find_root(node)
    if does_tree_import(u"__future__", feature, node):
        return
    shebang_encoding_idx = None
    for idx, node in enumerate(root.children):
        if is_shebang_comment(node) or is_encoding_comment(node):
            shebang_encoding_idx = idx
        if is_docstring(node):
            continue
        names = check_future_import(node)
        if not names:
            break
        if feature in names:
            return
    import_ = FromImport(u'__future__', [Leaf(token.NAME, feature, prefix=" ")])
    if shebang_encoding_idx == 0 and idx == 0:
        import_.prefix = root.children[0].prefix
        root.children[0].prefix = u''
    children = [import_ , Newline()]
    root.insert_child(idx, Node(syms.simple_stmt, children))
def future_import2(feature, node):
    root = find_root(node)
    if does_tree_import(u"__future__", feature, node):
        return
    insert_pos = 0
    for idx, node in enumerate(root.children):
        if node.type == syms.simple_stmt and node.children and           node.children[0].type == token.STRING:
            insert_pos = idx + 1
            break
    for thing_after in root.children[insert_pos:]:
        if thing_after.type == token.NEWLINE:
            insert_pos += 1
            continue
        prefix = thing_after.prefix
        thing_after.prefix = u""
        break
    else:
        prefix = u""
    import_ = FromImport(u"__future__", [Leaf(token.NAME, feature, prefix=u" ")])
    children = [import_, Newline()]
    root.insert_child(insert_pos, Node(syms.simple_stmt, children, prefix=prefix))
def parse_args(arglist, scheme):
    arglist = [i for i in arglist if i.type != token.COMMA]
    ret_mapping = dict([(k, None) for k in scheme])
    for i, arg in enumerate(arglist):
        if arg.type == syms.argument and arg.children[1].type == token.EQUAL:
            slot = arg.children[0].value
            ret_mapping[slot] = arg.children[2]
        else:
            slot = scheme[i]
            ret_mapping[slot] = arg
    return ret_mapping
def is_import_stmt(node):
    return (node.type == syms.simple_stmt and node.children and
            is_import(node.children[0]))
def touch_import_top(package, name_to_import, node):
    root = find_root(node)
    if does_tree_import(package, name_to_import, root):
        return
    found = False
    for name in ['absolute_import', 'division', 'print_function',
                 'unicode_literals']:
        if does_tree_import('__future__', name, root):
            found = True
            break
    if found:
        start, end = None, None
        for idx, node in enumerate(root.children):
            if check_future_import(node):
                start = idx
                idx2 = start
                while node:
                    node = node.next_sibling
                    idx2 += 1
                    if not check_future_import(node):
                        end = idx2
                        break
                break
        assert start is not None
        assert end is not None
        insert_pos = end
    else:
        for idx, node in enumerate(root.children):
            if node.type != syms.simple_stmt:
                break
            if not is_docstring(node):
                break
        insert_pos = idx
    if package is None:
        import_ = Node(syms.import_name, [
            Leaf(token.NAME, u"import"),
            Leaf(token.NAME, name_to_import, prefix=u" ")
        ])
    else:
        import_ = FromImport(package, [Leaf(token.NAME, name_to_import, prefix=u" ")])
        if name_to_import == u'standard_library':
            install_hooks = Node(syms.simple_stmt,
                                 [Node(syms.power,
                                       [Leaf(token.NAME, u'standard_library'),
                                        Node(syms.trailer, [Leaf(token.DOT, u'.'),
                                        Leaf(token.NAME, u'install_aliases')]),
                                        Node(syms.trailer, [Leaf(token.LPAR, u'('),
                                                            Leaf(token.RPAR, u')')])
                                       ])
                                 ]
                                )
            children_hooks = [install_hooks, Newline()]
        else:
            children_hooks = []
    children_import = [import_, Newline()]
    old_prefix = root.children[insert_pos].prefix
    root.children[insert_pos].prefix = u''
    root.insert_child(insert_pos, Node(syms.simple_stmt, children_import, prefix=old_prefix))
    if len(children_hooks) > 0:
        root.insert_child(insert_pos + 1, Node(syms.simple_stmt, children_hooks))
def check_future_import(node):
    savenode = node
    if not (node.type == syms.simple_stmt and node.children):
        return set()
    node = node.children[0]
    if not (node.type == syms.import_from and
            hasattr(node.children[1], 'value') and
            node.children[1].value == u'__future__'):
        return set()
    if node.children[3].type == token.LPAR:
        node = node.children[4]
    else:
        node = node.children[3]
    if node.type == syms.import_as_names:
        result = set()
        for n in node.children:
            if n.type == token.NAME:
                result.add(n.value)
            elif n.type == syms.import_as_name:
                n = n.children[0]
                assert n.type == token.NAME
                result.add(n.value)
        return result
    elif node.type == syms.import_as_name:
        node = node.children[0]
        assert node.type == token.NAME
        return set([node.value])
    elif node.type == token.NAME:
        return set([node.value])
    else:
        assert False, "strange import: %s" % savenode
def is_shebang_comment(node):
    return bool(re.match(SHEBANG_REGEX, node.prefix))
def is_encoding_comment(node):
    return bool(re.match(ENCODING_REGEX, node.prefix))
def wrap_in_fn_call(fn_name, args, prefix=None):
    assert len(args) > 0
    if len(args) == 2:
        expr1, expr2 = args
        newargs = [expr1, Comma(), expr2]
    else:
        newargs = args
    return Call(Name(fn_name), newargs, prefix=prefix)
