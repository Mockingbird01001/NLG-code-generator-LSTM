import types
import os
import sys
import comtypes.client
import comtypes.tools.codegenerator
import importlib
import logging
logger = logging.getLogger(__name__)
PATH = os.environ["PATH"].split(os.pathsep)
def _my_import(fullname):
    import comtypes.gen
    if comtypes.client.gen_dir           and comtypes.client.gen_dir not in comtypes.gen.__path__:
        comtypes.gen.__path__.append(comtypes.client.gen_dir)
    return __import__(fullname, globals(), locals(), ['DUMMY'])
def _name_module(tlib):
    libattr = tlib.GetLibAttr()
    modname = "_%s_%s_%s_%s" %              (str(libattr.guid)[1:-1].replace("-", "_"),
               libattr.lcid,
               libattr.wMajorVerNum,
               libattr.wMinorVerNum)
    return "comtypes.gen." + modname
def GetModule(tlib):
    pathname = None
    if isinstance(tlib, str):
        if not os.path.isabs(tlib):
            frame = sys._getframe(1)
            _file_ = frame.f_globals.get("__file__", None)
            if _file_ is not None:
                directory = os.path.dirname(os.path.abspath(_file_))
                abspath = os.path.normpath(os.path.join(directory, tlib))
                if os.path.isfile(abspath):
                    tlib = abspath
        logger.debug("GetModule(%s)", tlib)
        pathname = tlib
        tlib = comtypes.typeinfo.LoadTypeLibEx(tlib)
    elif isinstance(tlib, comtypes.GUID):
        clsid = str(tlib)
        import winreg
        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r"CLSID\%s\TypeLib" % clsid, 0, winreg.KEY_READ) as key:
            typelib = winreg.EnumValue(key, 0)[1]
        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r"CLSID\%s\Version" % clsid, 0, winreg.KEY_READ) as key:
            version = winreg.EnumValue(key, 0)[1].split(".")
        logger.debug("GetModule(%s)", typelib)
        tlib = comtypes.typeinfo.LoadRegTypeLib(comtypes.GUID(typelib), int(version[0]), int(version[1]), 0)
    elif isinstance(tlib, (tuple, list)):
        logger.debug("GetModule(%s)", (tlib,))
        tlib = comtypes.typeinfo.LoadRegTypeLib(comtypes.GUID(tlib[0]), *tlib[1:])
    elif hasattr(tlib, "_reg_libid_"):
        logger.debug("GetModule(%s)", tlib)
        tlib = comtypes.typeinfo.LoadRegTypeLib(comtypes.GUID(tlib._reg_libid_),
                                                *tlib._reg_version_)
    else:
        logger.debug("GetModule(%s)", tlib.GetLibAttr())
    mod = _CreateWrapper(tlib, pathname)
    try:
        modulename = tlib.GetDocumentation(-1)[0]
    except comtypes.COMError:
        return mod
    if modulename is None:
        return mod
    if sys.version_info < (3, 0):
        modulename = modulename.encode("mbcs")
    try:
        mod = _my_import("comtypes.gen." + modulename)
    except Exception as details:
        logger.info("Could not import comtypes.gen.%s: %s", modulename, details)
    else:
        return mod
    fullname = _name_module(tlib)
    modname = fullname.split(".")[-1]
    code = "from comtypes.gen import %s\nglobals().update(%s.__dict__)\n" % (modname, modname)
    code += "__name__ = 'comtypes.gen.%s'" % modulename
    if comtypes.client.gen_dir is None:
        mod = types.ModuleType("comtypes.gen." + modulename)
        mod.__file__ = os.path.join(os.path.abspath(comtypes.gen.__path__[0]),
                                    "<memory>")
        exec(code, mod.__dict__)
        sys.modules["comtypes.gen." + modulename] = mod
        setattr(comtypes.gen, modulename, mod)
        return mod
    ofi = open(os.path.join(comtypes.client.gen_dir, modulename + ".py"), "w")
    ofi.write(code)
    ofi.close()
    if hasattr(importlib, "invalidate_caches"):
        importlib.invalidate_caches()
    return _my_import("comtypes.gen." + modulename)
def _CreateWrapper(tlib, pathname=None):
    fullname = _name_module(tlib)
    try:
        return sys.modules[fullname]
    except KeyError:
        pass
    modname = fullname.split(".")[-1]
    try:
        return _my_import(fullname)
    except Exception as details:
        logger.info("Could not import %s: %s", fullname, details)
    from comtypes.tools.tlbparser import generate_module
    if comtypes.client.gen_dir is None:
        import io
        ofi = io.StringIO()
    else:
        ofi = open(os.path.join(comtypes.client.gen_dir, modname + ".py"), "w")
    generate_module(tlib, ofi, pathname)
    if comtypes.client.gen_dir is None:
        code = ofi.getvalue()
        mod = types.ModuleType(fullname)
        mod.__file__ = os.path.join(os.path.abspath(comtypes.gen.__path__[0]),
                                    "<memory>")
        exec(code, mod.__dict__)
        sys.modules[fullname] = mod
        setattr(comtypes.gen, modname, mod)
    else:
        ofi.close()
        if hasattr(importlib, "invalidate_caches"):
            importlib.invalidate_caches()
        mod = _my_import(fullname)
    return mod
if __name__ == "__main__":
    GetModule(sys.argv[1])
