
import functools
import sys
from .numerictypes import (
    string_, unicode_, integer, int_, object_, bool_, character)
from .numeric import ndarray, compare_chararrays
from .numeric import array as narray
from numpy.core.multiarray import _vec_string
from numpy.core.overrides import set_module
from numpy.core import overrides
from numpy.compat import asbytes
import numpy
__all__ = [
    'equal', 'not_equal', 'greater_equal', 'less_equal',
    'greater', 'less', 'str_len', 'add', 'multiply', 'mod', 'capitalize',
    'center', 'count', 'decode', 'encode', 'endswith', 'expandtabs',
    'find', 'index', 'isalnum', 'isalpha', 'isdigit', 'islower', 'isspace',
    'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'partition',
    'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit',
    'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase',
    'title', 'translate', 'upper', 'zfill', 'isnumeric', 'isdecimal',
    'array', 'asarray'
    ]
_globalvar = 0
array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy.char')
def _use_unicode(*args):
    for x in args:
        if (isinstance(x, str) or
                issubclass(numpy.asarray(x).dtype.type, unicode_)):
            return unicode_
    return string_
def _to_string_or_unicode_array(result):
    return numpy.asarray(result.tolist())
def _clean_args(*args):
    newargs = []
    for chk in args:
        if chk is None:
            break
        newargs.append(chk)
    return newargs
def _get_num_chars(a):
    if issubclass(a.dtype.type, unicode_):
        return a.itemsize // 4
    return a.itemsize
def _binary_op_dispatcher(x1, x2):
    return (x1, x2)
@array_function_dispatch(_binary_op_dispatcher)
def equal(x1, x2):
    return compare_chararrays(x1, x2, '==', True)
@array_function_dispatch(_binary_op_dispatcher)
def not_equal(x1, x2):
    return compare_chararrays(x1, x2, '!=', True)
@array_function_dispatch(_binary_op_dispatcher)
def greater_equal(x1, x2):
    return compare_chararrays(x1, x2, '>=', True)
@array_function_dispatch(_binary_op_dispatcher)
def less_equal(x1, x2):
    return compare_chararrays(x1, x2, '<=', True)
@array_function_dispatch(_binary_op_dispatcher)
def greater(x1, x2):
    return compare_chararrays(x1, x2, '>', True)
@array_function_dispatch(_binary_op_dispatcher)
def less(x1, x2):
    return compare_chararrays(x1, x2, '<', True)
def _unary_op_dispatcher(a):
    return (a,)
@array_function_dispatch(_unary_op_dispatcher)
def str_len(a):
    return _vec_string(a, int_, '__len__')
@array_function_dispatch(_binary_op_dispatcher)
def add(x1, x2):
    arr1 = numpy.asarray(x1)
    arr2 = numpy.asarray(x2)
    out_size = _get_num_chars(arr1) + _get_num_chars(arr2)
    dtype = _use_unicode(arr1, arr2)
    return _vec_string(arr1, (dtype, out_size), '__add__', (arr2,))
def _multiply_dispatcher(a, i):
    return (a,)
@array_function_dispatch(_multiply_dispatcher)
def multiply(a, i):
    a_arr = numpy.asarray(a)
    i_arr = numpy.asarray(i)
    if not issubclass(i_arr.dtype.type, integer):
        raise ValueError("Can only multiply by integers")
    out_size = _get_num_chars(a_arr) * max(int(i_arr.max()), 0)
    return _vec_string(
        a_arr, (a_arr.dtype.type, out_size), '__mul__', (i_arr,))
def _mod_dispatcher(a, values):
    return (a, values)
@array_function_dispatch(_mod_dispatcher)
def mod(a, values):
    return _to_string_or_unicode_array(
        _vec_string(a, object_, '__mod__', (values,)))
@array_function_dispatch(_unary_op_dispatcher)
def capitalize(a):
    a_arr = numpy.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'capitalize')
def _center_dispatcher(a, width, fillchar=None):
    return (a,)
@array_function_dispatch(_center_dispatcher)
def center(a, width, fillchar=' '):
    a_arr = numpy.asarray(a)
    width_arr = numpy.asarray(width)
    size = int(numpy.max(width_arr.flat))
    if numpy.issubdtype(a_arr.dtype, numpy.string_):
        fillchar = asbytes(fillchar)
    return _vec_string(
        a_arr, (a_arr.dtype.type, size), 'center', (width_arr, fillchar))
def _count_dispatcher(a, sub, start=None, end=None):
    return (a,)
@array_function_dispatch(_count_dispatcher)
def count(a, sub, start=0, end=None):
    return _vec_string(a, int_, 'count', [sub, start] + _clean_args(end))
def _code_dispatcher(a, encoding=None, errors=None):
    return (a,)
@array_function_dispatch(_code_dispatcher)
def decode(a, encoding=None, errors=None):
    return _to_string_or_unicode_array(
        _vec_string(a, object_, 'decode', _clean_args(encoding, errors)))
@array_function_dispatch(_code_dispatcher)
def encode(a, encoding=None, errors=None):
    return _to_string_or_unicode_array(
        _vec_string(a, object_, 'encode', _clean_args(encoding, errors)))
def _endswith_dispatcher(a, suffix, start=None, end=None):
    return (a,)
@array_function_dispatch(_endswith_dispatcher)
def endswith(a, suffix, start=0, end=None):
    return _vec_string(
        a, bool_, 'endswith', [suffix, start] + _clean_args(end))
def _expandtabs_dispatcher(a, tabsize=None):
    return (a,)
@array_function_dispatch(_expandtabs_dispatcher)
def expandtabs(a, tabsize=8):
    return _to_string_or_unicode_array(
        _vec_string(a, object_, 'expandtabs', (tabsize,)))
@array_function_dispatch(_count_dispatcher)
def find(a, sub, start=0, end=None):
    return _vec_string(
        a, int_, 'find', [sub, start] + _clean_args(end))
@array_function_dispatch(_count_dispatcher)
def index(a, sub, start=0, end=None):
    return _vec_string(
        a, int_, 'index', [sub, start] + _clean_args(end))
@array_function_dispatch(_unary_op_dispatcher)
def isalnum(a):
    return _vec_string(a, bool_, 'isalnum')
@array_function_dispatch(_unary_op_dispatcher)
def isalpha(a):
    return _vec_string(a, bool_, 'isalpha')
@array_function_dispatch(_unary_op_dispatcher)
def isdigit(a):
    return _vec_string(a, bool_, 'isdigit')
@array_function_dispatch(_unary_op_dispatcher)
def islower(a):
    return _vec_string(a, bool_, 'islower')
@array_function_dispatch(_unary_op_dispatcher)
def isspace(a):
    return _vec_string(a, bool_, 'isspace')
@array_function_dispatch(_unary_op_dispatcher)
def istitle(a):
    return _vec_string(a, bool_, 'istitle')
@array_function_dispatch(_unary_op_dispatcher)
def isupper(a):
    return _vec_string(a, bool_, 'isupper')
def _join_dispatcher(sep, seq):
    return (sep, seq)
@array_function_dispatch(_join_dispatcher)
def join(sep, seq):
    return _to_string_or_unicode_array(
        _vec_string(sep, object_, 'join', (seq,)))
def _just_dispatcher(a, width, fillchar=None):
    return (a,)
@array_function_dispatch(_just_dispatcher)
def ljust(a, width, fillchar=' '):
    a_arr = numpy.asarray(a)
    width_arr = numpy.asarray(width)
    size = int(numpy.max(width_arr.flat))
    if numpy.issubdtype(a_arr.dtype, numpy.string_):
        fillchar = asbytes(fillchar)
    return _vec_string(
        a_arr, (a_arr.dtype.type, size), 'ljust', (width_arr, fillchar))
@array_function_dispatch(_unary_op_dispatcher)
def lower(a):
    a_arr = numpy.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'lower')
def _strip_dispatcher(a, chars=None):
    return (a,)
@array_function_dispatch(_strip_dispatcher)
def lstrip(a, chars=None):
    a_arr = numpy.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'lstrip', (chars,))
def _partition_dispatcher(a, sep):
    return (a,)
@array_function_dispatch(_partition_dispatcher)
def partition(a, sep):
    return _to_string_or_unicode_array(
        _vec_string(a, object_, 'partition', (sep,)))
def _replace_dispatcher(a, old, new, count=None):
    return (a,)
@array_function_dispatch(_replace_dispatcher)
def replace(a, old, new, count=None):
    return _to_string_or_unicode_array(
        _vec_string(
            a, object_, 'replace', [old, new] + _clean_args(count)))
@array_function_dispatch(_count_dispatcher)
def rfind(a, sub, start=0, end=None):
    return _vec_string(
        a, int_, 'rfind', [sub, start] + _clean_args(end))
@array_function_dispatch(_count_dispatcher)
def rindex(a, sub, start=0, end=None):
    return _vec_string(
        a, int_, 'rindex', [sub, start] + _clean_args(end))
@array_function_dispatch(_just_dispatcher)
def rjust(a, width, fillchar=' '):
    a_arr = numpy.asarray(a)
    width_arr = numpy.asarray(width)
    size = int(numpy.max(width_arr.flat))
    if numpy.issubdtype(a_arr.dtype, numpy.string_):
        fillchar = asbytes(fillchar)
    return _vec_string(
        a_arr, (a_arr.dtype.type, size), 'rjust', (width_arr, fillchar))
@array_function_dispatch(_partition_dispatcher)
def rpartition(a, sep):
    return _to_string_or_unicode_array(
        _vec_string(a, object_, 'rpartition', (sep,)))
def _split_dispatcher(a, sep=None, maxsplit=None):
    return (a,)
@array_function_dispatch(_split_dispatcher)
def rsplit(a, sep=None, maxsplit=None):
    return _vec_string(
        a, object_, 'rsplit', [sep] + _clean_args(maxsplit))
def _strip_dispatcher(a, chars=None):
    return (a,)
@array_function_dispatch(_strip_dispatcher)
def rstrip(a, chars=None):
    a_arr = numpy.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'rstrip', (chars,))
@array_function_dispatch(_split_dispatcher)
def split(a, sep=None, maxsplit=None):
    return _vec_string(
        a, object_, 'split', [sep] + _clean_args(maxsplit))
def _splitlines_dispatcher(a, keepends=None):
    return (a,)
@array_function_dispatch(_splitlines_dispatcher)
def splitlines(a, keepends=None):
    return _vec_string(
        a, object_, 'splitlines', _clean_args(keepends))
def _startswith_dispatcher(a, prefix, start=None, end=None):
    return (a,)
@array_function_dispatch(_startswith_dispatcher)
def startswith(a, prefix, start=0, end=None):
    return _vec_string(
        a, bool_, 'startswith', [prefix, start] + _clean_args(end))
@array_function_dispatch(_strip_dispatcher)
def strip(a, chars=None):
    a_arr = numpy.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'strip', _clean_args(chars))
@array_function_dispatch(_unary_op_dispatcher)
def swapcase(a):
    a_arr = numpy.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'swapcase')
@array_function_dispatch(_unary_op_dispatcher)
def title(a):
    a_arr = numpy.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'title')
def _translate_dispatcher(a, table, deletechars=None):
    return (a,)
@array_function_dispatch(_translate_dispatcher)
def translate(a, table, deletechars=None):
    a_arr = numpy.asarray(a)
    if issubclass(a_arr.dtype.type, unicode_):
        return _vec_string(
            a_arr, a_arr.dtype, 'translate', (table,))
    else:
        return _vec_string(
            a_arr, a_arr.dtype, 'translate', [table] + _clean_args(deletechars))
@array_function_dispatch(_unary_op_dispatcher)
def upper(a):
    a_arr = numpy.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'upper')
def _zfill_dispatcher(a, width):
    return (a,)
@array_function_dispatch(_zfill_dispatcher)
def zfill(a, width):
    a_arr = numpy.asarray(a)
    width_arr = numpy.asarray(width)
    size = int(numpy.max(width_arr.flat))
    return _vec_string(
        a_arr, (a_arr.dtype.type, size), 'zfill', (width_arr,))
@array_function_dispatch(_unary_op_dispatcher)
def isnumeric(a):
    if _use_unicode(a) != unicode_:
        raise TypeError("isnumeric is only available for Unicode strings and arrays")
    return _vec_string(a, bool_, 'isnumeric')
@array_function_dispatch(_unary_op_dispatcher)
def isdecimal(a):
    if _use_unicode(a) != unicode_:
        raise TypeError("isnumeric is only available for Unicode strings and arrays")
    return _vec_string(a, bool_, 'isdecimal')
@set_module('numpy')
class chararray(ndarray):
    def __new__(subtype, shape, itemsize=1, unicode=False, buffer=None,
                offset=0, strides=None, order='C'):
        global _globalvar
        if unicode:
            dtype = unicode_
        else:
            dtype = string_
        itemsize = int(itemsize)
        if isinstance(buffer, str):
            filler = buffer
            buffer = None
        else:
            filler = None
        _globalvar = 1
        if buffer is None:
            self = ndarray.__new__(subtype, shape, (dtype, itemsize),
                                   order=order)
        else:
            self = ndarray.__new__(subtype, shape, (dtype, itemsize),
                                   buffer=buffer,
                                   offset=offset, strides=strides,
                                   order=order)
        if filler is not None:
            self[...] = filler
        _globalvar = 0
        return self
    def __array_finalize__(self, obj):
        if not _globalvar and self.dtype.char not in 'SUbc':
            raise ValueError("Can only create a chararray from string data.")
    def __getitem__(self, obj):
        val = ndarray.__getitem__(self, obj)
        if isinstance(val, character):
            temp = val.rstrip()
            if len(temp) == 0:
                val = ''
            else:
                val = temp
        return val
    def __eq__(self, other):
        return equal(self, other)
    def __ne__(self, other):
        return not_equal(self, other)
    def __ge__(self, other):
        return greater_equal(self, other)
    def __le__(self, other):
        return less_equal(self, other)
    def __gt__(self, other):
        return greater(self, other)
    def __lt__(self, other):
        return less(self, other)
    def __add__(self, other):
        return asarray(add(self, other))
    def __radd__(self, other):
        return asarray(add(numpy.asarray(other), self))
    def __mul__(self, i):
        return asarray(multiply(self, i))
    def __rmul__(self, i):
        return asarray(multiply(self, i))
    def __mod__(self, i):
        return asarray(mod(self, i))
    def __rmod__(self, other):
        return NotImplemented
    def argsort(self, axis=-1, kind=None, order=None):
        return self.__array__().argsort(axis, kind, order)
    argsort.__doc__ = ndarray.argsort.__doc__
    def capitalize(self):
        return asarray(capitalize(self))
    def center(self, width, fillchar=' '):
        return asarray(center(self, width, fillchar))
    def count(self, sub, start=0, end=None):
        return count(self, sub, start, end)
    def decode(self, encoding=None, errors=None):
        return decode(self, encoding, errors)
    def encode(self, encoding=None, errors=None):
        return encode(self, encoding, errors)
    def endswith(self, suffix, start=0, end=None):
        return endswith(self, suffix, start, end)
    def expandtabs(self, tabsize=8):
        return asarray(expandtabs(self, tabsize))
    def find(self, sub, start=0, end=None):
        return find(self, sub, start, end)
    def index(self, sub, start=0, end=None):
        return index(self, sub, start, end)
    def isalnum(self):
        return isalnum(self)
    def isalpha(self):
        return isalpha(self)
    def isdigit(self):
        return isdigit(self)
    def islower(self):
        return islower(self)
    def isspace(self):
        return isspace(self)
    def istitle(self):
        return istitle(self)
    def isupper(self):
        return isupper(self)
    def join(self, seq):
        return join(self, seq)
    def ljust(self, width, fillchar=' '):
        return asarray(ljust(self, width, fillchar))
    def lower(self):
        return asarray(lower(self))
    def lstrip(self, chars=None):
        return asarray(lstrip(self, chars))
    def partition(self, sep):
        return asarray(partition(self, sep))
    def replace(self, old, new, count=None):
        return asarray(replace(self, old, new, count))
    def rfind(self, sub, start=0, end=None):
        return rfind(self, sub, start, end)
    def rindex(self, sub, start=0, end=None):
        return rindex(self, sub, start, end)
    def rjust(self, width, fillchar=' '):
        return asarray(rjust(self, width, fillchar))
    def rpartition(self, sep):
        return asarray(rpartition(self, sep))
    def rsplit(self, sep=None, maxsplit=None):
        return rsplit(self, sep, maxsplit)
    def rstrip(self, chars=None):
        return asarray(rstrip(self, chars))
    def split(self, sep=None, maxsplit=None):
        return split(self, sep, maxsplit)
    def splitlines(self, keepends=None):
        return splitlines(self, keepends)
    def startswith(self, prefix, start=0, end=None):
        return startswith(self, prefix, start, end)
    def strip(self, chars=None):
        return asarray(strip(self, chars))
    def swapcase(self):
        return asarray(swapcase(self))
    def title(self):
        return asarray(title(self))
    def translate(self, table, deletechars=None):
        return asarray(translate(self, table, deletechars))
    def upper(self):
        return asarray(upper(self))
    def zfill(self, width):
        return asarray(zfill(self, width))
    def isnumeric(self):
        return isnumeric(self)
    def isdecimal(self):
        return isdecimal(self)
def array(obj, itemsize=None, copy=True, unicode=None, order=None):
    if isinstance(obj, (bytes, str)):
        if unicode is None:
            if isinstance(obj, str):
                unicode = True
            else:
                unicode = False
        if itemsize is None:
            itemsize = len(obj)
        shape = len(obj) // itemsize
        return chararray(shape, itemsize=itemsize, unicode=unicode,
                         buffer=obj, order=order)
    if isinstance(obj, (list, tuple)):
        obj = numpy.asarray(obj)
    if isinstance(obj, ndarray) and issubclass(obj.dtype.type, character):
        if not isinstance(obj, chararray):
            obj = obj.view(chararray)
        if itemsize is None:
            itemsize = obj.itemsize
            if issubclass(obj.dtype.type, unicode_):
                itemsize //= 4
        if unicode is None:
            if issubclass(obj.dtype.type, unicode_):
                unicode = True
            else:
                unicode = False
        if unicode:
            dtype = unicode_
        else:
            dtype = string_
        if order is not None:
            obj = numpy.asarray(obj, order=order)
        if (copy or
                (itemsize != obj.itemsize) or
                (not unicode and isinstance(obj, unicode_)) or
                (unicode and isinstance(obj, string_))):
            obj = obj.astype((dtype, int(itemsize)))
        return obj
    if isinstance(obj, ndarray) and issubclass(obj.dtype.type, object):
        if itemsize is None:
            obj = obj.tolist()
    if unicode:
        dtype = unicode_
    else:
        dtype = string_
    if itemsize is None:
        val = narray(obj, dtype=dtype, order=order, subok=True)
    else:
        val = narray(obj, dtype=(dtype, itemsize), order=order, subok=True)
    return val.view(chararray)
def asarray(obj, itemsize=None, unicode=None, order=None):
    return array(obj, itemsize, copy=False,
                 unicode=unicode, order=order)
