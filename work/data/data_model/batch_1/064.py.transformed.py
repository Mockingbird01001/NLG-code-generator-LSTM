
from __future__ import absolute_import, print_function, unicode_literals
import argparse
import sys
from pip._vendor.chardet import __version__
from pip._vendor.chardet.compat import PY2
from pip._vendor.chardet.universaldetector import UniversalDetector
def description_of(lines, name='stdin'):
    u = UniversalDetector()
    for line in lines:
        line = bytearray(line)
        u.feed(line)
        if u.done:
            break
    u.close()
    result = u.result
    if PY2:
        name = name.decode(sys.getfilesystemencoding(), 'ignore')
    if result['encoding']:
        return '{}: {} with confidence {}'.format(name, result['encoding'],
                                                     result['confidence'])
    else:
        return '{}: no result'.format(name)
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Takes one or more file paths and reports their detected \
                     encodings")
    parser.add_argument('input',
                        help='File whose encoding we would like to determine. \
                              (default: stdin)',
                        type=argparse.FileType('rb'), nargs='*',
                        default=[sys.stdin if PY2 else sys.stdin.buffer])
    parser.add_argument('--version', action='version',
                        version='%(prog)s {}'.format(__version__))
    args = parser.parse_args(argv)
    for f in args.input:
        if f.isatty():
            print("You are running chardetect interactively. Press " +
                  "CTRL-D twice at the start of a blank line to signal the " +
                  "end of your input. If you want help, run chardetect " +
                  "--help\n", file=sys.stderr)
        print(description_of(f, f.name))
if __name__ == '__main__':
    main()
