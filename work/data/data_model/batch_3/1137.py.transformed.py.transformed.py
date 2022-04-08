
import ctypes
from comtypes.tools.typedesc_base import *
class TypeLib(object):
    def __init__(self, name, guid, major, minor, doc=None):
        self.name = name
        self.guid = guid
        self.major = major
        self.minor = minor
        self.doc = doc
    def __repr__(self):
        return "<TypeLib(%s: %s, %s, %s)>" % (self.name, self.guid, self.major, self.minor)
class Constant(object):
    def __init__(self, name, typ, value):
        self.name = name
        self.typ = typ
        self.value = value
class External(object):
    def __init__(self, tlib, name, size, align, docs=None):
        self.tlib = tlib
        self.symbol_name = name
        self.size = size
        self.align = align
        self.docs = docs
    def get_head(self):
        return self
class SAFEARRAYType(object):
    def __init__(self, typ):
        self.typ = typ
        self.align = self.size = ctypes.sizeof(ctypes.c_void_p) * 8
class ComMethod(object):
    def __init__(self, invkind, memid, name, returns, idlflags, doc):
        self.invkind = invkind
        self.name = name
        self.returns = returns
        self.idlflags = idlflags
        self.memid = memid
        self.doc = doc
        self.arguments = []
    def add_argument(self, typ, name, idlflags, default):
        self.arguments.append((typ, name, idlflags, default))
class DispMethod(object):
    def __init__(self, dispid, invkind, name, returns, idlflags, doc):
        self.dispid = dispid
        self.invkind = invkind
        self.name = name
        self.returns = returns
        self.idlflags = idlflags
        self.doc = doc
        self.arguments = []
    def add_argument(self, typ, name, idlflags, default):
        self.arguments.append((typ, name, idlflags, default))
class DispProperty(object):
    def __init__(self, dispid, name, typ, idlflags, doc):
        self.dispid = dispid
        self.name = name
        self.typ = typ
        self.idlflags = idlflags
        self.doc = doc
class DispInterfaceHead(object):
    def __init__(self, itf):
        self.itf = itf
class DispInterfaceBody(object):
    def __init__(self, itf):
        self.itf = itf
class DispInterface(object):
    def __init__(self, name, members, base, iid, idlflags):
        self.name = name
        self.members = members
        self.base = base
        self.iid = iid
        self.idlflags = idlflags
        self.itf_head = DispInterfaceHead(self)
        self.itf_body = DispInterfaceBody(self)
    def get_body(self):
        return self.itf_body
    def get_head(self):
        return self.itf_head
class ComInterfaceHead(object):
    def __init__(self, itf):
        self.itf = itf
class ComInterfaceBody(object):
    def __init__(self, itf):
        self.itf = itf
class ComInterface(object):
    def __init__(self, name, members, base, iid, idlflags):
        self.name = name
        self.members = members
        self.base = base
        self.iid = iid
        self.idlflags = idlflags
        self.itf_head = ComInterfaceHead(self)
        self.itf_body = ComInterfaceBody(self)
    def get_body(self):
        return self.itf_body
    def get_head(self):
        return self.itf_head
class CoClass(object):
    def __init__(self, name, clsid, idlflags, tlibattr):
        self.name = name
        self.clsid = clsid
        self.idlflags = idlflags
        self.tlibattr = tlibattr
        self.interfaces = []
    def add_interface(self, itf, idlflags):
        self.interfaces.append((itf, idlflags))
