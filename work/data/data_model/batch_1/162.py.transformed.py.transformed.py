
import ctypes, logging, os, sys, tempfile, types
from ctypes import wintypes
logger = logging.getLogger(__name__)
def _ensure_list(path):
    return list(path)
def _find_gen_dir():
    _create_comtypes_gen_package()
    from comtypes import gen
    gen_path = _ensure_list(gen.__path__)
    if not _is_writeable(gen_path):
        ftype = getattr(sys, "frozen", None)
        version_str = "%d%d" % sys.version_info[:2]
        if ftype == None:
            subdir = r"Python\Python%s\comtypes_cache" % version_str
            basedir = _get_appdata_dir()
        elif ftype == "dll":
            path = _get_module_filename(sys.frozendllhandle)
            base = os.path.splitext(os.path.basename(path))[0]
            subdir = r"comtypes_cache\%s-%s" % (base, version_str)
            basedir = tempfile.gettempdir()
        else:
            base = os.path.splitext(os.path.basename(sys.executable))[0]
            subdir = r"comtypes_cache\%s-%s" % (base, version_str)
            basedir = tempfile.gettempdir()
        gen_dir = os.path.join(basedir, subdir)
        if not os.path.exists(gen_dir):
            logger.info("Creating writeable comtypes cache directory: '%s'", gen_dir)
            os.makedirs(gen_dir)
        gen_path.append(gen_dir)
    result = os.path.abspath(gen_path[-1])
    logger.info("Using writeable comtypes cache directory: '%s'", result)
    return result
SHGetSpecialFolderPath = ctypes.OleDLL("shell32.dll").SHGetSpecialFolderPathW
GetModuleFileName = ctypes.WinDLL("kernel32.dll").GetModuleFileNameW
SHGetSpecialFolderPath.argtypes = [ctypes.c_ulong, ctypes.c_wchar_p,
                                   ctypes.c_int, ctypes.c_int]
GetModuleFileName.restype = ctypes.c_ulong
GetModuleFileName.argtypes = [wintypes.HMODULE, ctypes.c_wchar_p, ctypes.c_ulong]
CSIDL_APPDATA = 26
MAX_PATH = 260
def _create_comtypes_gen_package():
    try:
        import comtypes.gen
        logger.info("Imported existing %s", comtypes.gen)
    except ImportError:
        import comtypes
        logger.info("Could not import comtypes.gen, trying to create it.")
        try:
            comtypes_path = os.path.abspath(os.path.join(comtypes.__path__[0], "gen"))
            if not os.path.isdir(comtypes_path):
                os.mkdir(comtypes_path)
                logger.info("Created comtypes.gen directory: '%s'", comtypes_path)
            comtypes_init = os.path.join(comtypes_path, "__init__.py")
            if not os.path.exists(comtypes_init):
                logger.info("Writing __init__.py file: '%s'", comtypes_init)
                ofi = open(comtypes_init, "w")
                ofi.close()
        except (OSError, IOError) as details:
            logger.info("Creating comtypes.gen package failed: %s", details)
            module = sys.modules["comtypes.gen"] = types.ModuleType("comtypes.gen")
            comtypes.gen = module
            comtypes.gen.__path__ = []
            logger.info("Created a memory-only package.")
def _is_writeable(path):
    if not path:
        return False
    return os.access(path[0], os.W_OK)
def _get_module_filename(hmodule):
    path = ctypes.create_unicode_buffer(MAX_PATH)
    if GetModuleFileName(hmodule, path, MAX_PATH):
        return path.value
    raise ctypes.WinError()
def _get_appdata_dir():
    path = ctypes.create_unicode_buffer(MAX_PATH)
    SHGetSpecialFolderPath(0, path, CSIDL_APPDATA, True)
    return path.value
