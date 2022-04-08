from __future__ import print_function
import os.path
import sys
from ..wheelfile import WheelFile
def unpack(path, dest='.'):
    with WheelFile(path) as wf:
        namever = wf.parsed_filename.group('namever')
        destination = os.path.join(dest, namever)
        print("Unpacking to: {}...".format(destination), end='')
        sys.stdout.flush()
        wf.extractall(destination)
    print('OK')
