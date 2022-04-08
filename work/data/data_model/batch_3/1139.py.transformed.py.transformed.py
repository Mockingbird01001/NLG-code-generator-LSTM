
import os
import sys
import weakref
from ctypes import *
from ctypes.wintypes import ULONG
from comtypes import STDMETHOD
from comtypes import COMMETHOD
from comtypes import _GUID, GUID
from comtypes.automation import BSTR
from comtypes.automation import DISPID
from comtypes.automation import DISPPARAMS
from comtypes.automation import DWORD
from comtypes.automation import EXCEPINFO
from comtypes.automation import HRESULT
from comtypes.automation import IID
from comtypes.automation import IUnknown
from comtypes.automation import LCID
from comtypes.automation import LONG
from comtypes.automation import SCODE
from comtypes.automation import UINT
from comtypes.automation import VARIANT
from comtypes.automation import VARIANTARG
from comtypes.automation import VARTYPE
from comtypes.automation import WCHAR
from comtypes.automation import WORD
from comtypes.automation import tagVARIANT
is_64_bit = sys.maxsize > 2**32
BOOL = c_int
HREFTYPE = DWORD
INT = c_int
MEMBERID = DISPID
OLECHAR = WCHAR
PVOID = c_void_p
SHORT = c_short
ULONG_PTR = c_uint64 if is_64_bit else c_ulong
USHORT = c_ushort
LPOLESTR = POINTER(OLECHAR)
tagSYSKIND = c_int
SYS_WIN16 = 0
SYS_WIN32 = 1
SYS_MAC = 2
SYS_WIN64 = 3
SYSKIND = tagSYSKIND
tagREGKIND = c_int
REGKIND_DEFAULT = 0
REGKIND_REGISTER = 1
REGKIND_NONE = 2
REGKIND = tagREGKIND
tagTYPEKIND = c_int
TKIND_ENUM = 0
TKIND_RECORD = 1
TKIND_MODULE = 2
TKIND_INTERFACE = 3
TKIND_DISPATCH = 4
TKIND_COCLASS = 5
TKIND_ALIAS = 6
TKIND_UNION = 7
TKIND_MAX = 8
TYPEKIND = tagTYPEKIND
tagINVOKEKIND = c_int
INVOKE_FUNC = 1
INVOKE_PROPERTYGET = 2
INVOKE_PROPERTYPUT = 4
INVOKE_PROPERTYPUTREF = 8
INVOKEKIND = tagINVOKEKIND
tagDESCKIND = c_int
DESCKIND_NONE = 0
DESCKIND_FUNCDESC = 1
DESCKIND_VARDESC = 2
DESCKIND_TYPECOMP = 3
DESCKIND_IMPLICITAPPOBJ = 4
DESCKIND_MAX = 5
DESCKIND = tagDESCKIND
tagVARKIND = c_int
VAR_PERINSTANCE = 0
VAR_STATIC = 1
VAR_CONST = 2
VAR_DISPATCH = 3
VARKIND = tagVARKIND
tagFUNCKIND = c_int
FUNC_VIRTUAL = 0
FUNC_PUREVIRTUAL = 1
FUNC_NONVIRTUAL = 2
FUNC_STATIC = 3
FUNC_DISPATCH = 4
FUNCKIND = tagFUNCKIND
tagCALLCONV = c_int
CC_FASTCALL = 0
CC_CDECL = 1
CC_MSCPASCAL = 2
CC_PASCAL = 2
CC_MACPASCAL = 3
CC_STDCALL = 4
CC_FPFASTCALL = 5
CC_SYSCALL = 6
CC_MPWCDECL = 7
CC_MPWPASCAL = 8
CC_MAX = 9
CALLCONV = tagCALLCONV
IMPLTYPEFLAG_FDEFAULT = 1
IMPLTYPEFLAG_FSOURCE = 2
IMPLTYPEFLAG_FRESTRICTED = 4
IMPLTYPEFLAG_FDEFAULTVTABLE = 8
tagTYPEFLAGS = c_int
TYPEFLAG_FAPPOBJECT = 1
TYPEFLAG_FCANCREATE = 2
TYPEFLAG_FLICENSED = 4
TYPEFLAG_FPREDECLID = 8
TYPEFLAG_FHIDDEN = 16
TYPEFLAG_FCONTROL = 32
TYPEFLAG_FDUAL = 64
TYPEFLAG_FNONEXTENSIBLE = 128
TYPEFLAG_FOLEAUTOMATION = 256
TYPEFLAG_FRESTRICTED = 512
TYPEFLAG_FAGGREGATABLE = 1024
TYPEFLAG_FREPLACEABLE = 2048
TYPEFLAG_FDISPATCHABLE = 4096
TYPEFLAG_FREVERSEBIND = 8192
TYPEFLAG_FPROXY = 16384
TYPEFLAGS = tagTYPEFLAGS
tagFUNCFLAGS = c_int
FUNCFLAG_FRESTRICTED = 1
FUNCFLAG_FSOURCE = 2
FUNCFLAG_FBINDABLE = 4
FUNCFLAG_FREQUESTEDIT = 8
FUNCFLAG_FDISPLAYBIND = 16
FUNCFLAG_FDEFAULTBIND = 32
FUNCFLAG_FHIDDEN = 64
FUNCFLAG_FUSESGETLASTERROR = 128
FUNCFLAG_FDEFAULTCOLLELEM = 256
FUNCFLAG_FUIDEFAULT = 512
FUNCFLAG_FNONBROWSABLE = 1024
FUNCFLAG_FREPLACEABLE = 2048
FUNCFLAG_FIMMEDIATEBIND = 4096
FUNCFLAGS = tagFUNCFLAGS
tagVARFLAGS = c_int
VARFLAG_FREADONLY = 1
VARFLAG_FSOURCE = 2
VARFLAG_FBINDABLE = 4
VARFLAG_FREQUESTEDIT = 8
VARFLAG_FDISPLAYBIND = 16
VARFLAG_FDEFAULTBIND = 32
VARFLAG_FHIDDEN = 64
VARFLAG_FRESTRICTED = 128
VARFLAG_FDEFAULTCOLLELEM = 256
VARFLAG_FUIDEFAULT = 512
VARFLAG_FNONBROWSABLE = 1024
VARFLAG_FREPLACEABLE = 2048
VARFLAG_FIMMEDIATEBIND = 4096
VARFLAGS = tagVARFLAGS
PARAMFLAG_NONE = 0
PARAMFLAG_FIN = 1
PARAMFLAG_FOUT = 2
PARAMFLAG_FLCID = 4
PARAMFLAG_FRETVAL = 8
PARAMFLAG_FOPT = 16
PARAMFLAG_FHASDEFAULT = 32
PARAMFLAG_FHASCUSTDATA = 64
def _deref_with_release(ptr, release):
    result = ptr[0]
    result.__ref__ = weakref.ref(result, lambda dead: release(ptr))
    return result
class ITypeLib(IUnknown):
    _iid_ = GUID("{00020402-0000-0000-C000-000000000046}")
    def GetLibAttr(self):
        return _deref_with_release(self._GetLibAttr(), self.ReleaseTLibAttr)
    def IsName(self, name, lHashVal=0):
        from ctypes import create_unicode_buffer
        namebuf = create_unicode_buffer(name)
        found = BOOL()
        self.__com_IsName(namebuf, lHashVal, byref(found))
        if found.value:
            return namebuf[:].split("\0", 1)[0]
        return None
    def FindName(self, name, lHashVal=0):
        found = c_ushort(1)
        tinfo = POINTER(ITypeInfo)()
        memid = MEMBERID()
        self.__com_FindName(name, lHashVal, byref(tinfo), byref(memid), byref(found))
        if found.value:
            return memid.value, tinfo
        return None
def fix_name(name):
    if name is None:
        return name
    return name.split("\0")[0]
class ITypeInfo(IUnknown):
    _iid_ = GUID("{00020401-0000-0000-C000-000000000046}")
    def GetTypeAttr(self):
        return _deref_with_release(self._GetTypeAttr(), self.ReleaseTypeAttr)
    def GetDocumentation(self, memid):
        name, doc, helpcontext, helpfile = self._GetDocumentation(memid)
        return fix_name(name), fix_name(doc), helpcontext, fix_name(helpfile)
    def GetFuncDesc(self, index):
        return _deref_with_release(self._GetFuncDesc(index), self.ReleaseFuncDesc)
    def GetVarDesc(self, index):
        return _deref_with_release(self._GetVarDesc(index), self.ReleaseVarDesc)
    def GetNames(self, memid, count=1):
        names = (BSTR * count)()
        cnames = c_uint()
        self.__com_GetNames(memid, names, count, byref(cnames))
        return names[:cnames.value]
    def GetIDsOfNames(self, *names):
        rgsznames = (c_wchar_p * len(names))(*names)
        ids = (MEMBERID * len(names))()
        self.__com_GetIDsOfNames(rgsznames, len(names), ids)
        return ids[:]
    def AddressOfMember(self, memid, invkind):
        raise "Check Me"
        p = c_void_p()
        self.__com_AddressOfMember(memid, invkind, byref(p))
        return p.value
    def CreateInstance(self, punkouter=None, interface=IUnknown, iid=None):
        if iid is None:
            iid = interface._iid_
        return self._CreateInstance(punkouter, byref(interface._iid_))
class ITypeComp(IUnknown):
    _iid_ = GUID("{00020403-0000-0000-C000-000000000046}")
    def Bind(self, name, flags=0, lHashVal=0):
        bindptr = BINDPTR()
        desckind = DESCKIND()
        ti = POINTER(ITypeInfo)()
        self.__com_Bind(name, lHashVal, flags, byref(ti), byref(desckind), byref(bindptr))
        kind = desckind.value
        if kind == DESCKIND_FUNCDESC:
            fd = bindptr.lpfuncdesc[0]
            fd.__ref__ = weakref.ref(fd, lambda dead: ti.ReleaseFuncDesc(bindptr.lpfuncdesc))
            return "function", fd
        elif kind == DESCKIND_VARDESC:
            vd = bindptr.lpvardesc[0]
            vd.__ref__ = weakref.ref(vd, lambda dead: ti.ReleaseVarDesc(bindptr.lpvardesc))
            return "variable", vd
        elif kind == DESCKIND_TYPECOMP:
            return "type", bindptr.lptcomp
        elif kind == DESCKIND_IMPLICITAPPOBJ:
            raise NotImplementedError
        elif kind == DESCKIND_NONE:
            raise NameError("Name %s not found" % name)
    def BindType(self, name, lHashVal=0):
        ti = POINTER(ITypeInfo)()
        tc = POINTER(ITypeComp)()
        self.__com_BindType(name, lHashVal, byref(ti), byref(tc))
        return ti, tc
class ICreateTypeLib(IUnknown):
    _iid_ = GUID("{00020406-0000-0000-C000-000000000046}")
class ICreateTypeLib2(ICreateTypeLib):
    _iid_ = GUID("{0002040F-0000-0000-C000-000000000046}")
class ICreateTypeInfo(IUnknown):
    _iid_ = GUID("{00020405-0000-0000-C000-000000000046}")
    def SetFuncAndParamNames(self, index, *names):
        rgszNames = (c_wchar_p * len(names))()
        for i, n in enumerate(names):
            rgszNames[i] = n
        return self._SetFuncAndParamNames(index, rgszNames, len(names))
class IRecordInfo(IUnknown):
    _iid_ = GUID("{0000002F-0000-0000-C000-000000000046}")
    def GetFieldNames(self, *args):
        count = c_ulong()
        self.__com_GetFieldNames(count, None)
        array = (BSTR * count.value)()
        self.__com_GetFieldNames(count, array)
        result = array[:]
        return result
IRecordInfo. _methods_ = [
        COMMETHOD([], HRESULT, 'RecordInit',
                  (['in'], c_void_p, 'pvNew')),
        COMMETHOD([], HRESULT, 'RecordClear',
                  (['in'], c_void_p, 'pvExisting')),
        COMMETHOD([], HRESULT, 'RecordCopy',
                  (['in'], c_void_p, 'pvExisting'),
                  (['in'], c_void_p, 'pvNew')),
        COMMETHOD([], HRESULT, 'GetGuid',
                  (['out'], POINTER(GUID), 'pguid')),
        COMMETHOD([], HRESULT, 'GetName',
                  (['out'], POINTER(BSTR), 'pbstrName')),
        COMMETHOD([], HRESULT, 'GetSize',
                  (['out'], POINTER(c_ulong), 'pcbSize')),
        COMMETHOD([], HRESULT, 'GetTypeInfo',
                  (['out'], POINTER(POINTER(ITypeInfo)), 'ppTypeInfo')),
        COMMETHOD([], HRESULT, 'GetField',
                  (['in'], c_void_p, 'pvData'),
                  (['in'], c_wchar_p, 'szFieldName'),
                  (['out'], POINTER(VARIANT), 'pvarField')),
        COMMETHOD([], HRESULT, 'GetFieldNoCopy',
                  (['in'], c_void_p, 'pvData'),
                  (['in'], c_wchar_p, 'szFieldName'),
                  (['out'], POINTER(VARIANT), 'pvarField'),
                  (['out'], POINTER(c_void_p), 'ppvDataCArray')),
        COMMETHOD([], HRESULT, 'PutField',
                  (['in'], c_ulong, 'wFlags'),
                  (['in'], c_void_p, 'pvData'),
                  (['in'], c_wchar_p, 'szFieldName'),
                  (['in'], POINTER(VARIANT), 'pvarField')),
        COMMETHOD([], HRESULT, 'PutFieldNoCopy',
                  (['in'], c_ulong, 'wFlags'),
                  (['in'], c_void_p, 'pvData'),
                  (['in'], c_wchar_p, 'szFieldName'),
                  (['in'], POINTER(VARIANT), 'pvarField')),
        COMMETHOD([], HRESULT, 'GetFieldNames',
                  (['in', 'out'], POINTER(c_ulong), 'pcNames'),
                  (['in'], POINTER(BSTR), 'rgBstrNames')),
        COMMETHOD([], BOOL, 'IsMatchingType',
                  (['in'], POINTER(IRecordInfo))),
        COMMETHOD([], HRESULT, 'RecordCreate'),
        COMMETHOD([], HRESULT, 'RecordCreateCopy',
                  (['in'], c_void_p, 'pvSource'),
                  (['out'], POINTER(c_void_p), 'ppvDest')),
        COMMETHOD([], HRESULT, 'RecordDestroy',
                  (['in'], c_void_p, 'pvRecord'))]
_oleaut32 = oledll.oleaut32
def GetRecordInfoFromTypeInfo(tinfo):
    ri = POINTER(IRecordInfo)()
    _oleaut32.GetRecordInfoFromTypeInfo(tinfo, byref(ri))
    return ri
def GetRecordInfoFromGuids(rGuidTypeLib, verMajor, verMinor, lcid, rGuidTypeInfo):
    ri = POINTER(IRecordInfo)()
    _oleaut32.GetRecordInfoFromGuids(byref(GUID(rGuidTypeLib)),
                                     verMajor, verMinor, lcid,
                                     byref(GUID(rGuidTypeInfo)),
                                     byref(ri))
    return ri
def LoadRegTypeLib(guid, wMajorVerNum, wMinorVerNum, lcid=0):
    tlib = POINTER(ITypeLib)()
    _oleaut32.LoadRegTypeLib(byref(GUID(guid)), wMajorVerNum, wMinorVerNum, lcid, byref(tlib))
    return tlib
if hasattr(_oleaut32, "LoadTypeLibEx"):
    def LoadTypeLibEx(szFile, regkind=REGKIND_NONE):
        ptl = POINTER(ITypeLib)()
        _oleaut32.LoadTypeLibEx(c_wchar_p(szFile), regkind, byref(ptl))
        return ptl
else:
    def LoadTypeLibEx(szFile, regkind=REGKIND_NONE):
        ptl = POINTER(ITypeLib)()
        _oleaut32.LoadTypeLib(c_wchar_p(szFile), byref(ptl))
        return ptl
def LoadTypeLib(szFile):
    tlib = POINTER(ITypeLib)()
    _oleaut32.LoadTypeLib(c_wchar_p(szFile), byref(tlib))
    return tlib
def UnRegisterTypeLib(libID, wVerMajor, wVerMinor, lcid=0, syskind=SYS_WIN32):
    return _oleaut32.UnRegisterTypeLib(byref(GUID(libID)), wVerMajor, wVerMinor, lcid, syskind)
def RegisterTypeLib(tlib, fullpath, helpdir=None):
    return _oleaut32.RegisterTypeLib(tlib, c_wchar_p(fullpath), c_wchar_p(helpdir))
def CreateTypeLib(filename, syskind=SYS_WIN32):
    ctlib = POINTER(ICreateTypeLib2)()
    _oleaut32.CreateTypeLib2(syskind, c_wchar_p(filename), byref(ctlib))
    return ctlib
def QueryPathOfRegTypeLib(libid, wVerMajor, wVerMinor, lcid=0):
    pathname = BSTR()
    _oleaut32.QueryPathOfRegTypeLib(byref(GUID(libid)), wVerMajor, wVerMinor, lcid, byref(pathname))
    return pathname.value.split("\0")[0]
class tagTLIBATTR(Structure):
    def __repr__(self):
        return "TLIBATTR(GUID=%s, Version=%s.%s, LCID=%s, FLags=0x%x)" %               (self.guid, self.wMajorVerNum, self.wMinorVerNum, self.lcid, self.wLibFlags)
TLIBATTR = tagTLIBATTR
class tagTYPEATTR(Structure):
    def __repr__(self):
        return "TYPEATTR(GUID=%s, typekind=%s, funcs=%s, vars=%s, impltypes=%s)" %               (self.guid, self.typekind, self.cFuncs, self.cVars, self.cImplTypes)
TYPEATTR = tagTYPEATTR
class tagFUNCDESC(Structure):
    def __repr__(self):
        return "FUNCDESC(memid=%s, cParams=%s, cParamsOpt=%s, callconv=%s, invkind=%s, funckind=%s)" %               (self.memid, self.cParams, self.cParamsOpt, self.callconv, self.invkind, self.funckind)
FUNCDESC = tagFUNCDESC
class tagVARDESC(Structure):
    pass
VARDESC = tagVARDESC
class tagBINDPTR(Union):
    pass
BINDPTR = tagBINDPTR
class tagTYPEDESC(Structure):
    pass
TYPEDESC = tagTYPEDESC
class tagIDLDESC(Structure):
    pass
IDLDESC = tagIDLDESC
class tagARRAYDESC(Structure):
    pass
ICreateTypeLib._methods_ = [
    COMMETHOD([], HRESULT, 'CreateTypeInfo',
              (['in'], LPOLESTR, 'szName'),
              (['in'], TYPEKIND, 'tkind'),
              (['out'], POINTER(POINTER(ICreateTypeInfo)), 'ppCTInfo')),
    STDMETHOD(HRESULT, 'SetName', [LPOLESTR]),
    STDMETHOD(HRESULT, 'SetVersion', [WORD, WORD]),
    STDMETHOD(HRESULT, 'SetGuid', [POINTER(GUID)]),
    STDMETHOD(HRESULT, 'SetDocString', [LPOLESTR]),
    STDMETHOD(HRESULT, 'SetHelpFileName', [LPOLESTR]),
    STDMETHOD(HRESULT, 'SetHelpContext', [DWORD]),
    STDMETHOD(HRESULT, 'SetLcid', [LCID]),
    STDMETHOD(HRESULT, 'SetLibFlags', [UINT]),
    STDMETHOD(HRESULT, 'SaveAllChanges', []),
]
ICreateTypeLib2._methods_ = [
    STDMETHOD(HRESULT, 'DeleteTypeInfo', [POINTER(ITypeInfo)]),
    STDMETHOD(HRESULT, 'SetCustData', [POINTER(GUID), POINTER(VARIANT)]),
    STDMETHOD(HRESULT, 'SetHelpStringContext', [ULONG]),
    STDMETHOD(HRESULT, 'SetHelpStringDll', [LPOLESTR]),
    ]
ITypeLib._methods_ = [
    COMMETHOD([], UINT, 'GetTypeInfoCount'),
    COMMETHOD([], HRESULT, 'GetTypeInfo',
              (['in'], UINT, 'index'),
              (['out'], POINTER(POINTER(ITypeInfo)))),
    COMMETHOD([], HRESULT, 'GetTypeInfoType',
              (['in'], UINT, 'index'),
              (['out'], POINTER(TYPEKIND))),
    COMMETHOD([], HRESULT, 'GetTypeInfoOfGuid',
              (['in'], POINTER(GUID)),
              (['out'], POINTER(POINTER(ITypeInfo)))),
    COMMETHOD([], HRESULT, 'GetLibAttr',
              (['out'], POINTER(POINTER(TLIBATTR)))),
    COMMETHOD([], HRESULT, 'GetTypeComp',
              (['out'], POINTER(POINTER(ITypeComp)))),
    COMMETHOD([], HRESULT, 'GetDocumentation',
              (['in'], INT, 'index'),
              (['out'], POINTER(BSTR)),
              (['out'], POINTER(BSTR)),
              (['out'], POINTER(DWORD)),
              (['out'], POINTER(BSTR))),
    COMMETHOD([], HRESULT, 'IsName',
              (['in', 'out'], LPOLESTR, 'name'),
              (['in', 'optional'], DWORD, 'lHashVal', 0),
              (['out'], POINTER(BOOL))),
    STDMETHOD(HRESULT, 'FindName', [LPOLESTR, DWORD, POINTER(POINTER(ITypeInfo)),
                                    POINTER(MEMBERID), POINTER(USHORT)]),
    COMMETHOD([], None, 'ReleaseTLibAttr',
              (['in'], POINTER(TLIBATTR)))
]
ITypeInfo._methods_ = [
    COMMETHOD([], HRESULT, 'GetTypeAttr',
              (['out'], POINTER(POINTER(TYPEATTR)), 'ppTypeAttr')),
    COMMETHOD([], HRESULT, 'GetTypeComp',
              (['out'], POINTER(POINTER(ITypeComp)))),
    COMMETHOD([], HRESULT, 'GetFuncDesc',
              (['in'], UINT, 'index'),
              (['out'], POINTER(POINTER(FUNCDESC)))),
    COMMETHOD([], HRESULT, 'GetVarDesc',
              (['in'], UINT, 'index'),
              (['out'], POINTER(POINTER(VARDESC)))),
    STDMETHOD(HRESULT, 'GetNames', [MEMBERID, POINTER(BSTR), UINT, POINTER(UINT)]),
    COMMETHOD([], HRESULT, 'GetRefTypeOfImplType',
              (['in'], UINT, 'index'),
              (['out'], POINTER(HREFTYPE))),
    COMMETHOD([], HRESULT, 'GetImplTypeFlags',
              (['in'], UINT, 'index'),
              (['out'], POINTER(INT))),
    STDMETHOD(HRESULT, 'GetIDsOfNames', [POINTER(c_wchar_p), UINT, POINTER(MEMBERID)]),
    STDMETHOD(HRESULT, 'Invoke', [PVOID, MEMBERID, WORD, POINTER(DISPPARAMS), POINTER(VARIANT), POINTER(EXCEPINFO), POINTER(UINT)]),
    COMMETHOD([], HRESULT, 'GetDocumentation',
              (['in'], MEMBERID, 'memid'),
              (['out'], POINTER(BSTR), 'pBstrName'),
              (['out'], POINTER(BSTR), 'pBstrDocString'),
              (['out'], POINTER(DWORD), 'pdwHelpContext'),
              (['out'], POINTER(BSTR), 'pBstrHelpFile')),
    COMMETHOD([], HRESULT, 'GetDllEntry',
              (['in'], MEMBERID, 'index'),
              (['in'], INVOKEKIND, 'invkind'),
              (['out'], POINTER(BSTR), 'pBstrDllName'),
              (['out'], POINTER(BSTR), 'pBstrName'),
              (['out'], POINTER(WORD), 'pwOrdinal')),
    COMMETHOD([], HRESULT, 'GetRefTypeInfo',
              (['in'], HREFTYPE, 'hRefType'),
              (['out'], POINTER(POINTER(ITypeInfo)))),
    STDMETHOD(HRESULT, 'AddressOfMember', [MEMBERID, INVOKEKIND, POINTER(PVOID)]),
    COMMETHOD([], HRESULT, 'CreateInstance',
              (['in'], POINTER(IUnknown), 'pUnkOuter'),
              (['in'], POINTER(IID), 'refiid'),
              (['out'], POINTER(POINTER(IUnknown)))),
    COMMETHOD([], HRESULT, 'GetMops',
              (['in'], MEMBERID, 'memid'),
              (['out'], POINTER(BSTR))),
    COMMETHOD([], HRESULT, 'GetContainingTypeLib',
              (['out'], POINTER(POINTER(ITypeLib))),
              (['out'], POINTER(UINT))),
    COMMETHOD([], None, 'ReleaseTypeAttr',
              (['in'], POINTER(TYPEATTR))),
    COMMETHOD([], None, 'ReleaseFuncDesc',
              (['in'], POINTER(FUNCDESC))),
    COMMETHOD([], None, 'ReleaseVarDesc',
              (['in'], POINTER(VARDESC))),
]
ITypeComp._methods_ = [
    STDMETHOD(HRESULT, 'Bind',
              [LPOLESTR, DWORD, WORD, POINTER(POINTER(ITypeInfo)),
               POINTER(DESCKIND), POINTER(BINDPTR)]),
    STDMETHOD(HRESULT, 'BindType',
              [LPOLESTR, DWORD, POINTER(POINTER(ITypeInfo)), POINTER(POINTER(ITypeComp))]),
]
ICreateTypeInfo._methods_ = [
    STDMETHOD(HRESULT, 'SetGuid', [POINTER(GUID)]),
    STDMETHOD(HRESULT, 'SetTypeFlags', [UINT]),
    STDMETHOD(HRESULT, 'SetDocString', [LPOLESTR]),
    STDMETHOD(HRESULT, 'SetHelpContext', [DWORD]),
    STDMETHOD(HRESULT, 'SetVersion', [WORD, WORD]),
    COMMETHOD([], HRESULT, 'AddRefTypeInfo',
              (['in'], POINTER(ITypeInfo)),
              (['out'], POINTER(HREFTYPE))),
    STDMETHOD(HRESULT, 'AddFuncDesc', [UINT, POINTER(FUNCDESC)]),
    STDMETHOD(HRESULT, 'AddImplType', [UINT, HREFTYPE]),
    STDMETHOD(HRESULT, 'SetImplTypeFlags', [UINT, INT]),
    STDMETHOD(HRESULT, 'SetAlignment', [WORD]),
    STDMETHOD(HRESULT, 'SetSchema', [LPOLESTR]),
    STDMETHOD(HRESULT, 'AddVarDesc', [UINT, POINTER(VARDESC)]),
    STDMETHOD(HRESULT, 'SetFuncAndParamNames', [UINT, POINTER(c_wchar_p), UINT]),
    STDMETHOD(HRESULT, 'SetVarName', [UINT, LPOLESTR]),
    STDMETHOD(HRESULT, 'SetTypeDescAlias', [POINTER(TYPEDESC)]),
    STDMETHOD(HRESULT, 'DefineFuncAsDllEntry', [UINT, LPOLESTR, LPOLESTR]),
    STDMETHOD(HRESULT, 'SetFuncDocString', [UINT, LPOLESTR]),
    STDMETHOD(HRESULT, 'SetVarDocString', [UINT, LPOLESTR]),
    STDMETHOD(HRESULT, 'SetFuncHelpContext', [UINT, DWORD]),
    STDMETHOD(HRESULT, 'SetVarHelpContext', [UINT, DWORD]),
    STDMETHOD(HRESULT, 'SetMops', [UINT, BSTR]),
    STDMETHOD(HRESULT, 'SetTypeIdldesc', [POINTER(IDLDESC)]),
    STDMETHOD(HRESULT, 'LayOut', []),
]
class IProvideClassInfo(IUnknown):
    _iid_ = GUID("{B196B283-BAB4-101A-B69C-00AA00341D07}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetClassInfo",
                  ( ['out'],  POINTER(POINTER(ITypeInfo)), "ppTI" ) )
        ]
class IProvideClassInfo2(IProvideClassInfo):
    _iid_ = GUID("{A6BC3AC0-DBAA-11CE-9DE3-00AA004BB851}")
    _methods_ = [
        COMMETHOD([], HRESULT, "GetGUID",
                  ( ['in'], DWORD, "dwGuidKind" ),
                  ( ['out', 'retval'], POINTER(GUID), "pGUID" ))
        ]
tagTLIBATTR._fields_ = [
    ('guid', GUID),
    ('lcid', LCID),
    ('syskind', SYSKIND),
    ('wMajorVerNum', WORD),
    ('wMinorVerNum', WORD),
    ('wLibFlags', WORD),
]
class N11tagTYPEDESC5DOLLAR_203E(Union):
    pass
N11tagTYPEDESC5DOLLAR_203E._fields_ = [
    ('lptdesc', POINTER(tagTYPEDESC)),
    ('lpadesc', POINTER(tagARRAYDESC)),
    ('hreftype', HREFTYPE),
]
tagTYPEDESC._anonymous_ = ('_',)
tagTYPEDESC._fields_ = [
    ('_', N11tagTYPEDESC5DOLLAR_203E),
    ('vt', VARTYPE),
]
tagIDLDESC._fields_ = [
    ('dwReserved', ULONG_PTR),
    ('wIDLFlags', USHORT),
]
tagTYPEATTR._fields_ = [
    ('guid', GUID),
    ('lcid', LCID),
    ('dwReserved', DWORD),
    ('memidConstructor', MEMBERID),
    ('memidDestructor', MEMBERID),
    ('lpstrSchema', LPOLESTR),
    ('cbSizeInstance', DWORD),
    ('typekind', TYPEKIND),
    ('cFuncs', WORD),
    ('cVars', WORD),
    ('cImplTypes', WORD),
    ('cbSizeVft', WORD),
    ('cbAlignment', WORD),
    ('wTypeFlags', WORD),
    ('wMajorVerNum', WORD),
    ('wMinorVerNum', WORD),
    ('tdescAlias', TYPEDESC),
    ('idldescType', IDLDESC),
]
class N10tagVARDESC5DOLLAR_205E(Union):
    pass
N10tagVARDESC5DOLLAR_205E._fields_ = [
    ('oInst', DWORD),
    ('lpvarValue', POINTER(VARIANT)),
]
class tagELEMDESC(Structure):
    pass
class N11tagELEMDESC5DOLLAR_204E(Union):
    pass
class tagPARAMDESC(Structure):
    pass
class tagPARAMDESCEX(Structure):
    pass
LPPARAMDESCEX = POINTER(tagPARAMDESCEX)
tagPARAMDESC._fields_ = [
    ('pparamdescex', LPPARAMDESCEX),
    ('wParamFlags', USHORT),
]
PARAMDESC = tagPARAMDESC
N11tagELEMDESC5DOLLAR_204E._fields_ = [
    ('idldesc', IDLDESC),
    ('paramdesc', PARAMDESC),
]
tagELEMDESC._fields_ = [
    ('tdesc', TYPEDESC),
    ('_', N11tagELEMDESC5DOLLAR_204E),
]
ELEMDESC = tagELEMDESC
tagVARDESC._fields_ = [
    ('memid', MEMBERID),
    ('lpstrSchema', LPOLESTR),
    ('_', N10tagVARDESC5DOLLAR_205E),
    ('elemdescVar', ELEMDESC),
    ('wVarFlags', WORD),
    ('varkind', VARKIND),
]
tagBINDPTR._fields_ = [
    ('lpfuncdesc', POINTER(FUNCDESC)),
    ('lpvardesc', POINTER(VARDESC)),
    ('lptcomp', POINTER(ITypeComp)),
]
tagFUNCDESC._fields_ = [
    ('memid', MEMBERID),
    ('lprgscode', POINTER(SCODE)),
    ('lprgelemdescParam', POINTER(ELEMDESC)),
    ('funckind', FUNCKIND),
    ('invkind', INVOKEKIND),
    ('callconv', CALLCONV),
    ('cParams', SHORT),
    ('cParamsOpt', SHORT),
    ('oVft', SHORT),
    ('cScodes', SHORT),
    ('elemdescFunc', ELEMDESC),
    ('wFuncFlags', WORD),
]
tagPARAMDESCEX._fields_ = [
    ('cBytes', DWORD),
    ('varDefaultValue', VARIANTARG),
]
class tagSAFEARRAYBOUND(Structure):
    _fields_ = [
        ('cElements', DWORD),
        ('lLbound', LONG),
    ]
SAFEARRAYBOUND = tagSAFEARRAYBOUND
tagARRAYDESC._fields_ = [
    ('tdescElem', TYPEDESC),
    ('cDims', USHORT),
    ('rgbounds', SAFEARRAYBOUND * 1),
]
