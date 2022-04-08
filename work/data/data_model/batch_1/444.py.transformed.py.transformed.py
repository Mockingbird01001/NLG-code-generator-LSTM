
from lib2to3 import fixer_base
from itertools import count
from lib2to3.fixer_util import (Assign, Comma, Call, Newline, Name,
                                Number, token, syms, Node, Leaf)
from libfuturize.fixer_util import indentation, suitify, commatize
def assignment_source(num_pre, num_post, LISTNAME, ITERNAME):
    children = []
    pre = unicode(num_pre)
    post = unicode(num_post)
    if num_pre > 0:
        pre_part = Node(syms.power, [Name(LISTNAME), Node(syms.trailer, [Leaf(token.LSQB, u"["), Node(syms.subscript, [Leaf(token.COLON, u":"), Number(pre)]), Leaf(token.RSQB, u"]")])])
        children.append(pre_part)
        children.append(Leaf(token.PLUS, u"+", prefix=u" "))
    main_part = Node(syms.power, [Leaf(token.LSQB, u"[", prefix=u" "), Name(LISTNAME), Node(syms.trailer, [Leaf(token.LSQB, u"["), Node(syms.subscript, [Number(pre) if num_pre > 0 else Leaf(1, u""), Leaf(token.COLON, u":"), Node(syms.factor, [Leaf(token.MINUS, u"-"), Number(post)]) if num_post > 0 else Leaf(1, u"")]), Leaf(token.RSQB, u"]"), Leaf(token.RSQB, u"]")])])
    children.append(main_part)
    if num_post > 0:
        children.append(Leaf(token.PLUS, u"+", prefix=u" "))
        post_part = Node(syms.power, [Name(LISTNAME, prefix=u" "), Node(syms.trailer, [Leaf(token.LSQB, u"["), Node(syms.subscript, [Node(syms.factor, [Leaf(token.MINUS, u"-"), Number(post)]), Leaf(token.COLON, u":")]), Leaf(token.RSQB, u"]")])])
        children.append(post_part)
    source = Node(syms.arith_expr, children)
    return source
class FixUnpacking(fixer_base.BaseFix):
    PATTERN = u"""
    expl=expr_stmt< testlist_star_expr<
        pre=(any ',')*
            star_expr< '*' name=NAME >
        post=(',' any)* [','] > '=' source=any > |
    impl=for_stmt< 'for' lst=exprlist<
        pre=(any ',')*
            star_expr< '*' name=NAME >
        post=(',' any)* [','] > 'in' it=any ':' suite=any>"""
    def fix_explicit_context(self, node, results):
        pre, name, post, source = (results.get(n) for n in (u"pre", u"name", u"post", u"source"))
        pre = [n.clone() for n in pre if n.type == token.NAME]
        name.prefix = u" "
        post = [n.clone() for n in post if n.type == token.NAME]
        target = [n.clone() for n in commatize(pre + [name.clone()] + post)]
        target.append(Comma())
        source.prefix = u""
        setup_line = Assign(Name(self.LISTNAME), Call(Name(u"list"), [source.clone()]))
        power_line = Assign(target, assignment_source(len(pre), len(post), self.LISTNAME, self.ITERNAME))
        return setup_line, power_line
    def fix_implicit_context(self, node, results):
        pre, name, post, it = (results.get(n) for n in (u"pre", u"name", u"post", u"it"))
        pre = [n.clone() for n in pre if n.type == token.NAME]
        name.prefix = u" "
        post = [n.clone() for n in post if n.type == token.NAME]
        target = [n.clone() for n in commatize(pre + [name.clone()] + post)]
        target.append(Comma())
        source = it.clone()
        source.prefix = u""
        setup_line = Assign(Name(self.LISTNAME), Call(Name(u"list"), [Name(self.ITERNAME)]))
        power_line = Assign(target, assignment_source(len(pre), len(post), self.LISTNAME, self.ITERNAME))
        return setup_line, power_line
    def transform(self, node, results):
        self.LISTNAME = self.new_name(u"_3to2list")
        self.ITERNAME = self.new_name(u"_3to2iter")
        expl, impl = results.get(u"expl"), results.get(u"impl")
        if expl is not None:
            setup_line, power_line = self.fix_explicit_context(node, results)
            setup_line.prefix = expl.prefix
            power_line.prefix = indentation(expl.parent)
            setup_line.append_child(Newline())
            parent = node.parent
            i = node.remove()
            parent.insert_child(i, power_line)
            parent.insert_child(i, setup_line)
        elif impl is not None:
            setup_line, power_line = self.fix_implicit_context(node, results)
            suitify(node)
            suite = [k for k in node.children if k.type == syms.suite][0]
            setup_line.prefix = u""
            power_line.prefix = suite.children[1].value
            suite.children[2].prefix = indentation(suite.children[2])
            suite.insert_child(2, Newline())
            suite.insert_child(2, power_line)
            suite.insert_child(2, Newline())
            suite.insert_child(2, setup_line)
            results.get(u"lst").replace(Name(self.ITERNAME, prefix=u" "))
