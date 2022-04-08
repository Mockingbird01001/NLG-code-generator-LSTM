
import os
import sys
import stat
from os.path import abspath
import fnmatch
try:
    from collections.abc import Callable
except ImportError:
    from collections import Callable
import errno
from . import tarfile
try:
    import bz2
    _BZ2_SUPPORTED = True
except ImportError:
    _BZ2_SUPPORTED = False
try:
    from pwd import getpwnam
except ImportError:
    getpwnam = None
try:
    from grp import getgrnam
except ImportError:
    getgrnam = None
__all__ = ["copyfileobj", "copyfile", "copymode", "copystat", "copy", "copy2",
           "copytree", "move", "rmtree", "Error", "SpecialFileError",
           "ExecError", "make_archive", "get_archive_formats",
           "register_archive_format", "unregister_archive_format",
           "get_unpack_formats", "register_unpack_format",
           "unregister_unpack_format", "unpack_archive", "ignore_patterns"]
class Error(EnvironmentError):
    pass
class SpecialFileError(EnvironmentError):
class ExecError(EnvironmentError):
class ReadError(EnvironmentError):
class RegistryError(Exception):
try:
    WindowsError
except NameError:
    WindowsError = None
def copyfileobj(fsrc, fdst, length=16*1024):
    while 1:
        buf = fsrc.read(length)
        if not buf:
            break
        fdst.write(buf)
def _samefile(src, dst):
    if hasattr(os.path, 'samefile'):
        try:
            return os.path.samefile(src, dst)
        except OSError:
            return False
    return (os.path.normcase(os.path.abspath(src)) ==
            os.path.normcase(os.path.abspath(dst)))
def copyfile(src, dst):
    if _samefile(src, dst):
        raise Error("`%s` and `%s` are the same file" % (src, dst))
    for fn in [src, dst]:
        try:
            st = os.stat(fn)
        except OSError:
            pass
        else:
            if stat.S_ISFIFO(st.st_mode):
                raise SpecialFileError("`%s` is a named pipe" % fn)
    with open(src, 'rb') as fsrc:
        with open(dst, 'wb') as fdst:
            copyfileobj(fsrc, fdst)
def copymode(src, dst):
    if hasattr(os, 'chmod'):
        st = os.stat(src)
        mode = stat.S_IMODE(st.st_mode)
        os.chmod(dst, mode)
def copystat(src, dst):
    st = os.stat(src)
    mode = stat.S_IMODE(st.st_mode)
    if hasattr(os, 'utime'):
        os.utime(dst, (st.st_atime, st.st_mtime))
    if hasattr(os, 'chmod'):
        os.chmod(dst, mode)
    if hasattr(os, 'chflags') and hasattr(st, 'st_flags'):
        try:
            os.chflags(dst, st.st_flags)
        except OSError as why:
            if (not hasattr(errno, 'EOPNOTSUPP') or
                why.errno != errno.EOPNOTSUPP):
                raise
def copy(src, dst):
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    copyfile(src, dst)
    copymode(src, dst)
def copy2(src, dst):
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    copyfile(src, dst)
    copystat(src, dst)
def ignore_patterns(*patterns):
    def _ignore_patterns(path, names):
        ignored_names = []
        for pattern in patterns:
            ignored_names.extend(fnmatch.filter(names, pattern))
        return set(ignored_names)
    return _ignore_patterns
def copytree(src, dst, symlinks=False, ignore=None, copy_function=copy2,
             ignore_dangling_symlinks=False):
    names = os.listdir(src)
    if ignore is not None:
        ignored_names = ignore(src, names)
    else:
        ignored_names = set()
    os.makedirs(dst)
    errors = []
    for name in names:
        if name in ignored_names:
            continue
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        try:
            if os.path.islink(srcname):
                linkto = os.readlink(srcname)
                if symlinks:
                    os.symlink(linkto, dstname)
                else:
                    if not os.path.exists(linkto) and ignore_dangling_symlinks:
                        continue
                    copy_function(srcname, dstname)
            elif os.path.isdir(srcname):
                copytree(srcname, dstname, symlinks, ignore, copy_function)
            else:
                copy_function(srcname, dstname)
        except Error as err:
            errors.extend(err.args[0])
        except EnvironmentError as why:
            errors.append((srcname, dstname, str(why)))
    try:
        copystat(src, dst)
    except OSError as why:
        if WindowsError is not None and isinstance(why, WindowsError):
            pass
        else:
            errors.extend((src, dst, str(why)))
    if errors:
        raise Error(errors)
def rmtree(path, ignore_errors=False, onerror=None):
    if ignore_errors:
        def onerror(*args):
            pass
    elif onerror is None:
        def onerror(*args):
            raise
    try:
        if os.path.islink(path):
            raise OSError("Cannot call rmtree on a symbolic link")
    except OSError:
        onerror(os.path.islink, path, sys.exc_info())
        return
    names = []
    try:
        names = os.listdir(path)
    except os.error:
        onerror(os.listdir, path, sys.exc_info())
    for name in names:
        fullname = os.path.join(path, name)
        try:
            mode = os.lstat(fullname).st_mode
        except os.error:
            mode = 0
        if stat.S_ISDIR(mode):
            rmtree(fullname, ignore_errors, onerror)
        else:
            try:
                os.remove(fullname)
            except os.error:
                onerror(os.remove, fullname, sys.exc_info())
    try:
        os.rmdir(path)
    except os.error:
        onerror(os.rmdir, path, sys.exc_info())
def _basename(path):
    return os.path.basename(path.rstrip(os.path.sep))
def move(src, dst):
    real_dst = dst
    if os.path.isdir(dst):
        if _samefile(src, dst):
            os.rename(src, dst)
            return
        real_dst = os.path.join(dst, _basename(src))
        if os.path.exists(real_dst):
            raise Error("Destination path '%s' already exists" % real_dst)
    try:
        os.rename(src, real_dst)
    except OSError:
        if os.path.isdir(src):
            if _destinsrc(src, dst):
                raise Error("Cannot move a directory '%s' into itself '%s'." % (src, dst))
            copytree(src, real_dst, symlinks=True)
            rmtree(src)
        else:
            copy2(src, real_dst)
            os.unlink(src)
def _destinsrc(src, dst):
    src = abspath(src)
    dst = abspath(dst)
    if not src.endswith(os.path.sep):
        src += os.path.sep
    if not dst.endswith(os.path.sep):
        dst += os.path.sep
    return dst.startswith(src)
def _get_gid(name):
    if getgrnam is None or name is None:
        return None
    try:
        result = getgrnam(name)
    except KeyError:
        result = None
    if result is not None:
        return result[2]
    return None
def _get_uid(name):
    if getpwnam is None or name is None:
        return None
    try:
        result = getpwnam(name)
    except KeyError:
        result = None
    if result is not None:
        return result[2]
    return None
def _make_tarball(base_name, base_dir, compress="gzip", verbose=0, dry_run=0,
                  owner=None, group=None, logger=None):
    tar_compression = {'gzip': 'gz', None: ''}
    compress_ext = {'gzip': '.gz'}
    if _BZ2_SUPPORTED:
        tar_compression['bzip2'] = 'bz2'
        compress_ext['bzip2'] = '.bz2'
    if compress is not None and compress not in compress_ext:
        raise ValueError("bad value for 'compress', or compression format not "
                         "supported : {0}".format(compress))
    archive_name = base_name + '.tar' + compress_ext.get(compress, '')
    archive_dir = os.path.dirname(archive_name)
    if not os.path.exists(archive_dir):
        if logger is not None:
            logger.info("creating %s", archive_dir)
        if not dry_run:
            os.makedirs(archive_dir)
    if logger is not None:
        logger.info('Creating tar archive')
    uid = _get_uid(owner)
    gid = _get_gid(group)
    def _set_uid_gid(tarinfo):
        if gid is not None:
            tarinfo.gid = gid
            tarinfo.gname = group
        if uid is not None:
            tarinfo.uid = uid
            tarinfo.uname = owner
        return tarinfo
    if not dry_run:
        tar = tarfile.open(archive_name, 'w|%s' % tar_compression[compress])
        try:
            tar.add(base_dir, filter=_set_uid_gid)
        finally:
            tar.close()
    return archive_name
def _call_external_zip(base_dir, zip_filename, verbose=False, dry_run=False):
    if verbose:
        zipoptions = "-r"
    else:
        zipoptions = "-rq"
    from distutils.errors import DistutilsExecError
    from distutils.spawn import spawn
    try:
        spawn(["zip", zipoptions, zip_filename, base_dir], dry_run=dry_run)
    except DistutilsExecError:
        raise ExecError("unable to create zip file '%s': "
            "could neither import the 'zipfile' module nor "
            "find a standalone zip utility") % zip_filename
def _make_zipfile(base_name, base_dir, verbose=0, dry_run=0, logger=None):
    zip_filename = base_name + ".zip"
    archive_dir = os.path.dirname(base_name)
    if not os.path.exists(archive_dir):
        if logger is not None:
            logger.info("creating %s", archive_dir)
        if not dry_run:
            os.makedirs(archive_dir)
    try:
        import zipfile
    except ImportError:
        zipfile = None
    if zipfile is None:
        _call_external_zip(base_dir, zip_filename, verbose, dry_run)
    else:
        if logger is not None:
            logger.info("creating '%s' and adding '%s' to it",
                        zip_filename, base_dir)
        if not dry_run:
            zip = zipfile.ZipFile(zip_filename, "w",
                                  compression=zipfile.ZIP_DEFLATED)
            for dirpath, dirnames, filenames in os.walk(base_dir):
                for name in filenames:
                    path = os.path.normpath(os.path.join(dirpath, name))
                    if os.path.isfile(path):
                        zip.write(path, path)
                        if logger is not None:
                            logger.info("adding '%s'", path)
            zip.close()
    return zip_filename
_ARCHIVE_FORMATS = {
    'gztar': (_make_tarball, [('compress', 'gzip')], "gzip'ed tar-file"),
    'bztar': (_make_tarball, [('compress', 'bzip2')], "bzip2'ed tar-file"),
    'tar':   (_make_tarball, [('compress', None)], "uncompressed tar file"),
    'zip':   (_make_zipfile, [], "ZIP file"),
    }
if _BZ2_SUPPORTED:
    _ARCHIVE_FORMATS['bztar'] = (_make_tarball, [('compress', 'bzip2')],
                                "bzip2'ed tar-file")
def get_archive_formats():
    formats = [(name, registry[2]) for name, registry in
               _ARCHIVE_FORMATS.items()]
    formats.sort()
    return formats
def register_archive_format(name, function, extra_args=None, description=''):
    if extra_args is None:
        extra_args = []
    if not isinstance(function, Callable):
        raise TypeError('The %s object is not callable' % function)
    if not isinstance(extra_args, (tuple, list)):
        raise TypeError('extra_args needs to be a sequence')
    for element in extra_args:
        if not isinstance(element, (tuple, list)) or len(element) !=2:
            raise TypeError('extra_args elements are : (arg_name, value)')
    _ARCHIVE_FORMATS[name] = (function, extra_args, description)
def unregister_archive_format(name):
    del _ARCHIVE_FORMATS[name]
def make_archive(base_name, format, root_dir=None, base_dir=None, verbose=0,
                 dry_run=0, owner=None, group=None, logger=None):
    save_cwd = os.getcwd()
    if root_dir is not None:
        if logger is not None:
            logger.debug("changing into '%s'", root_dir)
        base_name = os.path.abspath(base_name)
        if not dry_run:
            os.chdir(root_dir)
    if base_dir is None:
        base_dir = os.curdir
    kwargs = {'dry_run': dry_run, 'logger': logger}
    try:
        format_info = _ARCHIVE_FORMATS[format]
    except KeyError:
        raise ValueError("unknown archive format '%s'" % format)
    func = format_info[0]
    for arg, val in format_info[1]:
        kwargs[arg] = val
    if format != 'zip':
        kwargs['owner'] = owner
        kwargs['group'] = group
    try:
        filename = func(base_name, base_dir, **kwargs)
    finally:
        if root_dir is not None:
            if logger is not None:
                logger.debug("changing back to '%s'", save_cwd)
            os.chdir(save_cwd)
    return filename
def get_unpack_formats():
    formats = [(name, info[0], info[3]) for name, info in
               _UNPACK_FORMATS.items()]
    formats.sort()
    return formats
def _check_unpack_options(extensions, function, extra_args):
    existing_extensions = {}
    for name, info in _UNPACK_FORMATS.items():
        for ext in info[0]:
            existing_extensions[ext] = name
    for extension in extensions:
        if extension in existing_extensions:
            msg = '%s is already registered for "%s"'
            raise RegistryError(msg % (extension,
                                       existing_extensions[extension]))
    if not isinstance(function, Callable):
        raise TypeError('The registered function must be a callable')
def register_unpack_format(name, extensions, function, extra_args=None,
                           description=''):
    if extra_args is None:
        extra_args = []
    _check_unpack_options(extensions, function, extra_args)
    _UNPACK_FORMATS[name] = extensions, function, extra_args, description
def unregister_unpack_format(name):
    del _UNPACK_FORMATS[name]
def _ensure_directory(path):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
def _unpack_zipfile(filename, extract_dir):
    try:
        import zipfile
    except ImportError:
        raise ReadError('zlib not supported, cannot unpack this archive.')
    if not zipfile.is_zipfile(filename):
        raise ReadError("%s is not a zip file" % filename)
    zip = zipfile.ZipFile(filename)
    try:
        for info in zip.infolist():
            name = info.filename
            if name.startswith('/') or '..' in name:
                continue
            target = os.path.join(extract_dir, *name.split('/'))
            if not target:
                continue
            _ensure_directory(target)
            if not name.endswith('/'):
                data = zip.read(info.filename)
                f = open(target, 'wb')
                try:
                    f.write(data)
                finally:
                    f.close()
                    del data
    finally:
        zip.close()
def _unpack_tarfile(filename, extract_dir):
    try:
        tarobj = tarfile.open(filename)
    except tarfile.TarError:
        raise ReadError(
            "%s is not a compressed or uncompressed tar file" % filename)
    try:
        tarobj.extractall(extract_dir)
    finally:
        tarobj.close()
_UNPACK_FORMATS = {
    'gztar': (['.tar.gz', '.tgz'], _unpack_tarfile, [], "gzip'ed tar-file"),
    'tar':   (['.tar'], _unpack_tarfile, [], "uncompressed tar file"),
    'zip':   (['.zip'], _unpack_zipfile, [], "ZIP file")
    }
if _BZ2_SUPPORTED:
    _UNPACK_FORMATS['bztar'] = (['.bz2'], _unpack_tarfile, [],
                                "bzip2'ed tar-file")
def _find_unpack_format(filename):
    for name, info in _UNPACK_FORMATS.items():
        for extension in info[0]:
            if filename.endswith(extension):
                return name
    return None
def unpack_archive(filename, extract_dir=None, format=None):
    if extract_dir is None:
        extract_dir = os.getcwd()
    if format is not None:
        try:
            format_info = _UNPACK_FORMATS[format]
        except KeyError:
            raise ValueError("Unknown unpack format '{0}'".format(format))
        func = format_info[1]
        func(filename, extract_dir, **dict(format_info[2]))
    else:
        format = _find_unpack_format(filename)
        if format is None:
            raise ReadError("Unknown archive format '{0}'".format(filename))
        func = _UNPACK_FORMATS[format][1]
        kwargs = dict(_UNPACK_FORMATS[format][2])
        func(filename, extract_dir, **kwargs)
