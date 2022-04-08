
import distutils.dist
import os.path
import re
import sys
import tempfile
import zipfile
from argparse import ArgumentParser
from glob import iglob
from shutil import rmtree
import wheel.bdist_wheel
from wheel.archive import archive_wheelfile
egg_info_re = re.compile(r'''(^|/)(?P<name>[^/]+?)-(?P<ver>.+?)
    (-(?P<pyver>.+?))?(-(?P<arch>.+?))?.egg-info(/|$)''', re.VERBOSE)
def parse_info(wininfo_name, egginfo_name):
    egginfo = None
    if egginfo_name:
        egginfo = egg_info_re.search(egginfo_name)
        if not egginfo:
            raise ValueError("Egg info filename %s is not valid" % (egginfo_name,))
    w_name, sep, rest = wininfo_name.partition('-')
    if not sep:
        raise ValueError("Installer filename %s is not valid" % (wininfo_name,))
    rest = rest[:-4]
    rest2, sep, w_pyver = rest.rpartition('-')
    if sep and w_pyver.startswith('py'):
        rest = rest2
        w_pyver = w_pyver.replace('.', '')
    else:
        w_pyver = 'py2.py3'
    w_ver, sep, w_arch = rest.rpartition('.')
    if not sep:
        raise ValueError("Installer filename %s is not valid" % (wininfo_name,))
    if egginfo:
        w_name = egginfo.group('name')
        w_ver = egginfo.group('ver')
    return dict(name=w_name, ver=w_ver, arch=w_arch, pyver=w_pyver)
def bdist_wininst2wheel(path, dest_dir=os.path.curdir):
    bdw = zipfile.ZipFile(path)
    egginfo_name = None
    for filename in bdw.namelist():
        if '.egg-info' in filename:
            egginfo_name = filename
            break
    info = parse_info(os.path.basename(path), egginfo_name)
    root_is_purelib = True
    for zipinfo in bdw.infolist():
        if zipinfo.filename.startswith('PLATLIB'):
            root_is_purelib = False
            break
    if root_is_purelib:
        paths = {'purelib': ''}
    else:
        paths = {'platlib': ''}
    dist_info = "%(name)s-%(ver)s" % info
    datadir = "%s.data/" % dist_info
    members = []
    egginfo_name = ''
    for zipinfo in bdw.infolist():
        key, basename = zipinfo.filename.split('/', 1)
        key = key.lower()
        basepath = paths.get(key, None)
        if basepath is None:
            basepath = datadir + key.lower() + '/'
        oldname = zipinfo.filename
        newname = basepath + basename
        zipinfo.filename = newname
        del bdw.NameToInfo[oldname]
        bdw.NameToInfo[newname] = zipinfo
        if newname:
            members.append(newname)
        if not egginfo_name:
            if newname.endswith('.egg-info'):
                egginfo_name = newname
            elif '.egg-info/' in newname:
                egginfo_name, sep, _ = newname.rpartition('/')
    dir = tempfile.mkdtemp(suffix="_b2w")
    bdw.extractall(dir, members)
    abi = 'none'
    pyver = info['pyver']
    arch = (info['arch'] or 'any').replace('.', '_').replace('-', '_')
    if root_is_purelib:
        arch = 'any'
    if arch != 'any':
        pyver = pyver.replace('py', 'cp')
    wheel_name = '-'.join((
                          dist_info,
                          pyver,
                          abi,
                          arch
                          ))
    if root_is_purelib:
        bw = wheel.bdist_wheel.bdist_wheel(distutils.dist.Distribution())
    else:
        bw = _bdist_wheel_tag(distutils.dist.Distribution())
    bw.root_is_pure = root_is_purelib
    bw.python_tag = pyver
    bw.plat_name_supplied = True
    bw.plat_name = info['arch'] or 'any'
    if not root_is_purelib:
        bw.full_tag_supplied = True
        bw.full_tag = (pyver, abi, arch)
    dist_info_dir = os.path.join(dir, '%s.dist-info' % dist_info)
    bw.egg2dist(os.path.join(dir, egginfo_name), dist_info_dir)
    bw.write_wheelfile(dist_info_dir, generator='wininst2wheel')
    bw.write_record(dir, dist_info_dir)
    archive_wheelfile(os.path.join(dest_dir, wheel_name), dir)
    rmtree(dir)
class _bdist_wheel_tag(wheel.bdist_wheel.bdist_wheel):
    full_tag_supplied = False
    full_tag = None
    def get_tag(self):
        if self.full_tag_supplied and self.full_tag is not None:
            return self.full_tag
        else:
            return super(_bdist_wheel_tag, self).get_tag()
def main():
    parser = ArgumentParser()
    parser.add_argument('installers', nargs='*', help="Installers to convert")
    parser.add_argument('--dest-dir', '-d', default=os.path.curdir,
                        help="Directory to store wheels (default %(default)s)")
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()
    for pat in args.installers:
        for installer in iglob(pat):
            if args.verbose:
                sys.stdout.write("{0}... ".format(installer))
            bdist_wininst2wheel(installer, args.dest_dir)
            if args.verbose:
                sys.stdout.write("OK\n")
if __name__ == "__main__":
    main()
