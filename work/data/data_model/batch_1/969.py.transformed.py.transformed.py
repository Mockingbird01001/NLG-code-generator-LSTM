
"""
    werkzeug.security
    ~~~~~~~~~~~~~~~~~
    Security related helpers such as secure password hashing tools.
    :copyright: (c) 2014 by the Werkzeug Team, see AUTHORS for more details.
    :license: BSD, see LICENSE for more details.
"""
import os
import hmac
import hashlib
import posixpath
import codecs
from struct import Struct
from random import SystemRandom
from operator import xor
from itertools import starmap
from werkzeug._compat import range_type, PY2, text_type, izip, to_bytes,    string_types, to_native
SALT_CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
DEFAULT_PBKDF2_ITERATIONS = 50000
_pack_int = Struct('>I').pack
_builtin_safe_str_cmp = getattr(hmac, 'compare_digest', None)
_sys_rng = SystemRandom()
_os_alt_seps = list(sep for sep in [os.path.sep, os.path.altsep]
                    if sep not in (None, '/'))
def _find_hashlib_algorithms():
    algos = getattr(hashlib, 'algorithms', None)
    if algos is None:
        algos = ('md5', 'sha1', 'sha224', 'sha256', 'sha384', 'sha512')
    rv = {}
    for algo in algos:
        func = getattr(hashlib, algo, None)
        if func is not None:
            rv[algo] = func
    return rv
_hash_funcs = _find_hashlib_algorithms()
def pbkdf2_hex(data, salt, iterations=DEFAULT_PBKDF2_ITERATIONS,
               keylen=None, hashfunc=None):
    rv = pbkdf2_bin(data, salt, iterations, keylen, hashfunc)
    return to_native(codecs.encode(rv, 'hex_codec'))
_has_native_pbkdf2 = hasattr(hashlib, 'pbkdf2_hmac')
def pbkdf2_bin(data, salt, iterations=DEFAULT_PBKDF2_ITERATIONS,
               keylen=None, hashfunc=None):
    if isinstance(hashfunc, string_types):
        hashfunc = _hash_funcs[hashfunc]
    elif not hashfunc:
        hashfunc = hashlib.sha256
    data = to_bytes(data)
    salt = to_bytes(salt)
    if _has_native_pbkdf2:
        _test_hash = hashfunc()
        if hasattr(_test_hash, 'name') and           _test_hash.name in _hash_funcs:
            return hashlib.pbkdf2_hmac(_test_hash.name,
                                       data, salt, iterations,
                                       keylen)
    mac = hmac.HMAC(data, None, hashfunc)
    if not keylen:
        keylen = mac.digest_size
    def _pseudorandom(x, mac=mac):
        h = mac.copy()
        h.update(x)
        return bytearray(h.digest())
    buf = bytearray()
    for block in range_type(1, -(-keylen // mac.digest_size) + 1):
        rv = u = _pseudorandom(salt + _pack_int(block))
        for i in range_type(iterations - 1):
            u = _pseudorandom(bytes(u))
            rv = bytearray(starmap(xor, izip(rv, u)))
        buf.extend(rv)
    return bytes(buf[:keylen])
def safe_str_cmp(a, b):
    if isinstance(a, text_type):
        a = a.encode('utf-8')
    if isinstance(b, text_type):
        b = b.encode('utf-8')
    if _builtin_safe_str_cmp is not None:
        return _builtin_safe_str_cmp(a, b)
    if len(a) != len(b):
        return False
    rv = 0
    if PY2:
        for x, y in izip(a, b):
            rv |= ord(x) ^ ord(y)
    else:
        for x, y in izip(a, b):
            rv |= x ^ y
    return rv == 0
def gen_salt(length):
    if length <= 0:
        raise ValueError('Salt length must be positive')
    return ''.join(_sys_rng.choice(SALT_CHARS) for _ in range_type(length))
def _hash_internal(method, salt, password):
    if method == 'plain':
        return password, method
    if isinstance(password, text_type):
        password = password.encode('utf-8')
    if method.startswith('pbkdf2:'):
        args = method[7:].split(':')
        if len(args) not in (1, 2):
            raise ValueError('Invalid number of arguments for PBKDF2')
        method = args.pop(0)
        iterations = args and int(args[0] or 0) or DEFAULT_PBKDF2_ITERATIONS
        is_pbkdf2 = True
        actual_method = 'pbkdf2:%s:%d' % (method, iterations)
    else:
        is_pbkdf2 = False
        actual_method = method
    hash_func = _hash_funcs.get(method)
    if hash_func is None:
        raise TypeError('invalid method %r' % method)
    if is_pbkdf2:
        if not salt:
            raise ValueError('Salt is required for PBKDF2')
        rv = pbkdf2_hex(password, salt, iterations,
                        hashfunc=hash_func)
    elif salt:
        if isinstance(salt, text_type):
            salt = salt.encode('utf-8')
        rv = hmac.HMAC(salt, password, hash_func).hexdigest()
    else:
        h = hash_func()
        h.update(password)
        rv = h.hexdigest()
    return rv, actual_method
def generate_password_hash(password, method='pbkdf2:sha256', salt_length=8):
    salt = method != 'plain' and gen_salt(salt_length) or ''
    h, actual_method = _hash_internal(method, salt, password)
    return '%s$%s$%s' % (actual_method, salt, h)
def check_password_hash(pwhash, password):
    if pwhash.count('$') < 2:
        return False
    method, salt, hashval = pwhash.split('$', 2)
    return safe_str_cmp(_hash_internal(method, salt, password)[0], hashval)
def safe_join(directory, *pathnames):
    parts = [directory]
    for filename in pathnames:
        if filename != '':
            filename = posixpath.normpath(filename)
        for sep in _os_alt_seps:
            if sep in filename:
                return None
        if os.path.isabs(filename) or           filename == '..' or           filename.startswith('../'):
            return None
        parts.append(filename)
    return posixpath.join(*parts)
