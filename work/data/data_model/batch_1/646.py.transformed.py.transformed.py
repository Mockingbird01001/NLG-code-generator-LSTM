
import ctypes
import os
import sys
FAT_MAGIC = 0xcafebabe
FAT_CIGAM = 0xbebafeca
FAT_MAGIC_64 = 0xcafebabf
FAT_CIGAM_64 = 0xbfbafeca
MH_MAGIC = 0xfeedface
MH_CIGAM = 0xcefaedfe
MH_MAGIC_64 = 0xfeedfacf
MH_CIGAM_64 = 0xcffaedfe
LC_VERSION_MIN_MACOSX = 0x24
LC_BUILD_VERSION = 0x32
CPU_TYPE_ARM64 = 0x0100000c
mach_header_fields = [
        ("magic", ctypes.c_uint32), ("cputype", ctypes.c_int),
        ("cpusubtype", ctypes.c_int), ("filetype", ctypes.c_uint32),
        ("ncmds", ctypes.c_uint32), ("sizeofcmds", ctypes.c_uint32),
        ("flags", ctypes.c_uint32)
    ]
mach_header_fields_64 = mach_header_fields + [("reserved", ctypes.c_uint32)]
fat_header_fields = [("magic", ctypes.c_uint32), ("nfat_arch", ctypes.c_uint32)]
fat_arch_fields = [
    ("cputype", ctypes.c_int), ("cpusubtype", ctypes.c_int),
    ("offset", ctypes.c_uint32), ("size", ctypes.c_uint32),
    ("align", ctypes.c_uint32)
]
fat_arch_64_fields = [
    ("cputype", ctypes.c_int), ("cpusubtype", ctypes.c_int),
    ("offset", ctypes.c_uint64), ("size", ctypes.c_uint64),
    ("align", ctypes.c_uint32), ("reserved", ctypes.c_uint32)
]
segment_base_fields = [("cmd", ctypes.c_uint32), ("cmdsize", ctypes.c_uint32)]
segment_command_fields = [
    ("cmd", ctypes.c_uint32), ("cmdsize", ctypes.c_uint32),
    ("segname", ctypes.c_char * 16), ("vmaddr", ctypes.c_uint32),
    ("vmsize", ctypes.c_uint32), ("fileoff", ctypes.c_uint32),
    ("filesize", ctypes.c_uint32), ("maxprot", ctypes.c_int),
    ("initprot", ctypes.c_int), ("nsects", ctypes.c_uint32),
    ("flags", ctypes.c_uint32),
    ]
segment_command_fields_64 = [
    ("cmd", ctypes.c_uint32), ("cmdsize", ctypes.c_uint32),
    ("segname", ctypes.c_char * 16), ("vmaddr", ctypes.c_uint64),
    ("vmsize", ctypes.c_uint64), ("fileoff", ctypes.c_uint64),
    ("filesize", ctypes.c_uint64), ("maxprot", ctypes.c_int),
    ("initprot", ctypes.c_int), ("nsects", ctypes.c_uint32),
    ("flags", ctypes.c_uint32),
    ]
version_min_command_fields = segment_base_fields +    [("version", ctypes.c_uint32), ("sdk", ctypes.c_uint32)]
build_version_command_fields = segment_base_fields +    [("platform", ctypes.c_uint32), ("minos", ctypes.c_uint32),
     ("sdk", ctypes.c_uint32), ("ntools", ctypes.c_uint32)]
def swap32(x):
    return (((x << 24) & 0xFF000000) |
            ((x << 8) & 0x00FF0000) |
            ((x >> 8) & 0x0000FF00) |
            ((x >> 24) & 0x000000FF))
def get_base_class_and_magic_number(lib_file, seek=None):
    if seek is None:
        seek = lib_file.tell()
    else:
        lib_file.seek(seek)
    magic_number = ctypes.c_uint32.from_buffer_copy(
        lib_file.read(ctypes.sizeof(ctypes.c_uint32))).value
    if magic_number in [FAT_CIGAM, FAT_CIGAM_64, MH_CIGAM, MH_CIGAM_64]:
        if sys.byteorder == "little":
            BaseClass = ctypes.BigEndianStructure
        else:
            BaseClass = ctypes.LittleEndianStructure
        magic_number = swap32(magic_number)
    else:
        BaseClass = ctypes.Structure
    lib_file.seek(seek)
    return BaseClass, magic_number
def read_data(struct_class, lib_file):
    return struct_class.from_buffer_copy(lib_file.read(
                        ctypes.sizeof(struct_class)))
def extract_macosx_min_system_version(path_to_lib):
    with open(path_to_lib, "rb") as lib_file:
        BaseClass, magic_number = get_base_class_and_magic_number(lib_file, 0)
        if magic_number not in [FAT_MAGIC, FAT_MAGIC_64, MH_MAGIC, MH_MAGIC_64]:
            return
        if magic_number in [FAT_MAGIC, FAT_CIGAM_64]:
            class FatHeader(BaseClass):
                _fields_ = fat_header_fields
            fat_header = read_data(FatHeader, lib_file)
            if magic_number == FAT_MAGIC:
                class FatArch(BaseClass):
                    _fields_ = fat_arch_fields
            else:
                class FatArch(BaseClass):
                    _fields_ = fat_arch_64_fields
            fat_arch_list = [read_data(FatArch, lib_file) for _ in range(fat_header.nfat_arch)]
            versions_list = []
            for el in fat_arch_list:
                try:
                    version = read_mach_header(lib_file, el.offset)
                    if version is not None:
                        if el.cputype == CPU_TYPE_ARM64 and len(fat_arch_list) != 1:
                            if version == (11, 0, 0):
                                continue
                        versions_list.append(version)
                except ValueError:
                    pass
            if len(versions_list) > 0:
                return max(versions_list)
            else:
                return None
        else:
            try:
                return read_mach_header(lib_file, 0)
            except ValueError:
                return None
def read_mach_header(lib_file, seek=None):
    if seek is not None:
        lib_file.seek(seek)
    base_class, magic_number = get_base_class_and_magic_number(lib_file)
    arch = "32" if magic_number == MH_MAGIC else "64"
    class SegmentBase(base_class):
        _fields_ = segment_base_fields
    if arch == "32":
        class MachHeader(base_class):
            _fields_ = mach_header_fields
    else:
        class MachHeader(base_class):
            _fields_ = mach_header_fields_64
    mach_header = read_data(MachHeader, lib_file)
    for _i in range(mach_header.ncmds):
        pos = lib_file.tell()
        segment_base = read_data(SegmentBase, lib_file)
        lib_file.seek(pos)
        if segment_base.cmd == LC_VERSION_MIN_MACOSX:
            class VersionMinCommand(base_class):
                _fields_ = version_min_command_fields
            version_info = read_data(VersionMinCommand, lib_file)
            return parse_version(version_info.version)
        elif segment_base.cmd == LC_BUILD_VERSION:
            class VersionBuild(base_class):
                _fields_ = build_version_command_fields
            version_info = read_data(VersionBuild, lib_file)
            return parse_version(version_info.minos)
        else:
            lib_file.seek(pos + segment_base.cmdsize)
            continue
def parse_version(version):
    x = (version & 0xffff0000) >> 16
    y = (version & 0x0000ff00) >> 8
    z = (version & 0x000000ff)
    return x, y, z
def calculate_macosx_platform_tag(archive_root, platform_tag):
    prefix, base_version, suffix = platform_tag.split('-')
    base_version = tuple([int(x) for x in base_version.split(".")])
    base_version = base_version[:2]
    if base_version[0] > 10:
        base_version = (base_version[0], 0)
    assert len(base_version) == 2
    if "MACOSX_DEPLOYMENT_TARGET" in os.environ:
        deploy_target = tuple([int(x) for x in os.environ[
            "MACOSX_DEPLOYMENT_TARGET"].split(".")])
        deploy_target = deploy_target[:2]
        if deploy_target[0] > 10:
            deploy_target = (deploy_target[0], 0)
        if deploy_target < base_version:
            sys.stderr.write(
                 "[WARNING] MACOSX_DEPLOYMENT_TARGET is set to a lower value ({}) than the "
                 "version on which the Python interpreter was compiled ({}), and will be "
                 "ignored.\n".format('.'.join(str(x) for x in deploy_target),
                                     '.'.join(str(x) for x in base_version))
                )
        else:
            base_version = deploy_target
    assert len(base_version) == 2
    start_version = base_version
    versions_dict = {}
    for (dirpath, dirnames, filenames) in os.walk(archive_root):
        for filename in filenames:
            if filename.endswith('.dylib') or filename.endswith('.so'):
                lib_path = os.path.join(dirpath, filename)
                min_ver = extract_macosx_min_system_version(lib_path)
                if min_ver is not None:
                    min_ver = min_ver[0:2]
                    if min_ver[0] > 10:
                        min_ver = (min_ver[0], 0)
                    versions_dict[lib_path] = min_ver
    if len(versions_dict) > 0:
        base_version = max(base_version, max(versions_dict.values()))
    fin_base_version = "_".join([str(x) for x in base_version])
    if start_version < base_version:
        problematic_files = [k for k, v in versions_dict.items() if v > start_version]
        problematic_files = "\n".join(problematic_files)
        if len(problematic_files) == 1:
            files_form = "this file"
        else:
            files_form = "these files"
        error_message =            "[WARNING] This wheel needs a higher macOS version than {}  "            "To silence this warning, set MACOSX_DEPLOYMENT_TARGET to at least " +            fin_base_version + " or recreate " + files_form + " with lower "            "MACOSX_DEPLOYMENT_TARGET:  \n" + problematic_files
        if "MACOSX_DEPLOYMENT_TARGET" in os.environ:
            error_message = error_message.format("is set in MACOSX_DEPLOYMENT_TARGET variable.")
        else:
            error_message = error_message.format(
                "the version your Python interpreter is compiled against.")
        sys.stderr.write(error_message)
    platform_tag = prefix + "_" + fin_base_version + "_" + suffix
    return platform_tag
