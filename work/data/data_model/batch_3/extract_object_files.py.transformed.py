
"""Module for extracting object files from a compiled archive (.a) file.
This module provides functionality almost identical to the 'ar -x' command,
which extracts out all object files from a given archive file. This module
assumes the archive is in the BSD variant format used in Apple platforms.
This extractor has two important differences compared to the 'ar -x' command
shipped with Xcode.
1.  When there are multiple object files with the same name in a given archive,
    each file is renamed so that they are all correctly extracted without
    overwriting each other.
2.  This module takes the destination directory as an additional parameter.
    Example Usage:
    archive_path = ...
    dest_dir = ...
    extract_object_files(archive_path, dest_dir)
"""
import hashlib
import io
import itertools
import os
import struct
from typing import Iterator, Tuple
def extract_object_files(archive_file: io.BufferedIOBase,
                         dest_dir: str) -> None:
  if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
  _check_archive_signature(archive_file)
  extracted_files = dict()
  for name, file_content in _extract_next_file(archive_file):
    digest = hashlib.md5(file_content).digest()
    for final_name in _generate_modified_filenames(name):
      if final_name not in extracted_files:
        extracted_files[final_name] = digest
        with open(os.path.join(dest_dir, final_name), 'wb') as object_file:
          object_file.write(file_content)
        break
      elif extracted_files[final_name] == digest:
        break
def _generate_modified_filenames(filename: str) -> Iterator[str]:
  yield filename
  base, ext = os.path.splitext(filename)
  for name_suffix in itertools.count(1, 1):
    yield '{}_{}{}'.format(base, name_suffix, ext)
def _check_archive_signature(archive_file: io.BufferedIOBase) -> None:
  signature = archive_file.read(8)
  if signature != b'!<arch>\n':
    raise RuntimeError('Invalid archive file format.')
def _extract_next_file(
    archive_file: io.BufferedIOBase) -> Iterator[Tuple[str, bytes]]:
  while True:
    header = archive_file.read(60)
    if not header:
      return
    elif len(header) < 60:
      raise RuntimeError('Invalid file header format.')
    name, _, _, _, _, size, end = struct.unpack('=16s12s6s6s8s10s2s', header)
    if end != b'`\n':
      raise RuntimeError('Invalid file header format.')
    name = name.decode('ascii').strip()
    size = int(size, base=10)
    odd_size = size % 2 == 1
      filename_size = int(name[3:])
      name = archive_file.read(filename_size).decode('utf-8').strip(' \x00')
      size -= filename_size
    file_content = archive_file.read(size)
    if odd_size:
      archive_file.read(1)
    yield (name, file_content)
