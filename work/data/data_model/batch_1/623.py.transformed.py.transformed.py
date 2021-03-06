
__all__ = ['matrix_power', 'solve', 'tensorsolve', 'tensorinv', 'inv',
           'cholesky', 'eigvals', 'eigvalsh', 'pinv', 'slogdet', 'det',
           'svd', 'eig', 'eigh', 'lstsq', 'norm', 'qr', 'cond', 'matrix_rank',
           'LinAlgError', 'multi_dot']
import functools
import operator
import warnings
from numpy.core import (
    array, asarray, zeros, empty, empty_like, intc, single, double,
    csingle, cdouble, inexact, complexfloating, newaxis, all, Inf, dot,
    add, multiply, sqrt, fastCopyAndTranspose, sum, isfinite,
    finfo, errstate, geterrobj, moveaxis, amin, amax, product, abs,
    atleast_2d, intp, asanyarray, object_, matmul,
    swapaxes, divide, count_nonzero, isnan, sign, argsort, sort
)
from numpy.core.multiarray import normalize_axis_index
from numpy.core.overrides import set_module
from numpy.core import overrides
from numpy.lib.twodim_base import triu, eye
from numpy.linalg import lapack_lite, _umath_linalg
array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy.linalg')
fortran_int = intc
@set_module('numpy.linalg')
class LinAlgError(Exception):
def _determine_error_states():
    errobj = geterrobj()
    bufsize = errobj[0]
    with errstate(invalid='call', over='ignore',
                  divide='ignore', under='ignore'):
        invalid_call_errmask = geterrobj()[1]
    return [bufsize, invalid_call_errmask, None]
_linalg_error_extobj = _determine_error_states()
del _determine_error_states
def _raise_linalgerror_singular(err, flag):
    raise LinAlgError("Singular matrix")
def _raise_linalgerror_nonposdef(err, flag):
    raise LinAlgError("Matrix is not positive definite")
def _raise_linalgerror_eigenvalues_nonconvergence(err, flag):
    raise LinAlgError("Eigenvalues did not converge")
def _raise_linalgerror_svd_nonconvergence(err, flag):
    raise LinAlgError("SVD did not converge")
def _raise_linalgerror_lstsq(err, flag):
    raise LinAlgError("SVD did not converge in Linear Least Squares")
def get_linalg_error_extobj(callback):
    extobj = list(_linalg_error_extobj)
    extobj[2] = callback
    return extobj
def _makearray(a):
    new = asarray(a)
    wrap = getattr(a, "__array_prepare__", new.__array_wrap__)
    return new, wrap
def isComplexType(t):
    return issubclass(t, complexfloating)
_real_types_map = {single : single,
                   double : double,
                   csingle : single,
                   cdouble : double}
_complex_types_map = {single : csingle,
                      double : cdouble,
                      csingle : csingle,
                      cdouble : cdouble}
def _realType(t, default=double):
    return _real_types_map.get(t, default)
def _complexType(t, default=cdouble):
    return _complex_types_map.get(t, default)
def _linalgRealType(t):
    return double
def _commonType(*arrays):
    result_type = single
    is_complex = False
    for a in arrays:
        if issubclass(a.dtype.type, inexact):
            if isComplexType(a.dtype.type):
                is_complex = True
            rt = _realType(a.dtype.type, default=None)
            if rt is None:
                raise TypeError("array type %s is unsupported in linalg" %
                        (a.dtype.name,))
        else:
            rt = double
        if rt is double:
            result_type = double
    if is_complex:
        t = cdouble
        result_type = _complex_types_map[result_type]
    else:
        t = double
    return t, result_type
_fastCT = fastCopyAndTranspose
def _to_native_byte_order(*arrays):
    ret = []
    for arr in arrays:
        if arr.dtype.byteorder not in ('=', '|'):
            ret.append(asarray(arr, dtype=arr.dtype.newbyteorder('=')))
        else:
            ret.append(arr)
    if len(ret) == 1:
        return ret[0]
    else:
        return ret
def _fastCopyAndTranspose(type, *arrays):
    cast_arrays = ()
    for a in arrays:
        if a.dtype.type is not type:
            a = a.astype(type)
        cast_arrays = cast_arrays + (_fastCT(a),)
    if len(cast_arrays) == 1:
        return cast_arrays[0]
    else:
        return cast_arrays
def _assert_2d(*arrays):
    for a in arrays:
        if a.ndim != 2:
            raise LinAlgError('%d-dimensional array given. Array must be '
                    'two-dimensional' % a.ndim)
def _assert_stacked_2d(*arrays):
    for a in arrays:
        if a.ndim < 2:
            raise LinAlgError('%d-dimensional array given. Array must be '
                    'at least two-dimensional' % a.ndim)
def _assert_stacked_square(*arrays):
    for a in arrays:
        m, n = a.shape[-2:]
        if m != n:
            raise LinAlgError('Last 2 dimensions of the array must be square')
def _assert_finite(*arrays):
    for a in arrays:
        if not isfinite(a).all():
            raise LinAlgError("Array must not contain infs or NaNs")
def _is_empty_2d(arr):
    return arr.size == 0 and product(arr.shape[-2:]) == 0
def transpose(a):
    return swapaxes(a, -1, -2)
def _tensorsolve_dispatcher(a, b, axes=None):
    return (a, b)
@array_function_dispatch(_tensorsolve_dispatcher)
def tensorsolve(a, b, axes=None):
    a, wrap = _makearray(a)
    b = asarray(b)
    an = a.ndim
    if axes is not None:
        allaxes = list(range(0, an))
        for k in axes:
            allaxes.remove(k)
            allaxes.insert(an, k)
        a = a.transpose(allaxes)
    oldshape = a.shape[-(an-b.ndim):]
    prod = 1
    for k in oldshape:
        prod *= k
    a = a.reshape(-1, prod)
    b = b.ravel()
    res = wrap(solve(a, b))
    res.shape = oldshape
    return res
def _solve_dispatcher(a, b):
    return (a, b)
@array_function_dispatch(_solve_dispatcher)
def solve(a, b):
    a, _ = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    b, wrap = _makearray(b)
    t, result_t = _commonType(a, b)
    if b.ndim == a.ndim - 1:
        gufunc = _umath_linalg.solve1
    else:
        gufunc = _umath_linalg.solve
    signature = 'DD->D' if isComplexType(t) else 'dd->d'
    extobj = get_linalg_error_extobj(_raise_linalgerror_singular)
    r = gufunc(a, b, signature=signature, extobj=extobj)
    return wrap(r.astype(result_t, copy=False))
def _tensorinv_dispatcher(a, ind=None):
    return (a,)
@array_function_dispatch(_tensorinv_dispatcher)
def tensorinv(a, ind=2):
    a = asarray(a)
    oldshape = a.shape
    prod = 1
    if ind > 0:
        invshape = oldshape[ind:] + oldshape[:ind]
        for k in oldshape[ind:]:
            prod *= k
    else:
        raise ValueError("Invalid ind argument.")
    a = a.reshape(prod, -1)
    ia = inv(a)
    return ia.reshape(*invshape)
def _unary_dispatcher(a):
    return (a,)
@array_function_dispatch(_unary_dispatcher)
def inv(a):
    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)
    signature = 'D->D' if isComplexType(t) else 'd->d'
    extobj = get_linalg_error_extobj(_raise_linalgerror_singular)
    ainv = _umath_linalg.inv(a, signature=signature, extobj=extobj)
    return wrap(ainv.astype(result_t, copy=False))
def _matrix_power_dispatcher(a, n):
    return (a,)
@array_function_dispatch(_matrix_power_dispatcher)
def matrix_power(a, n):
    a = asanyarray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    try:
        n = operator.index(n)
    except TypeError as e:
        raise TypeError("exponent must be an integer") from e
    if a.dtype != object:
        fmatmul = matmul
    elif a.ndim == 2:
        fmatmul = dot
    else:
        raise NotImplementedError(
            "matrix_power not supported for stacks of object arrays")
    if n == 0:
        a = empty_like(a)
        a[...] = eye(a.shape[-2], dtype=a.dtype)
        return a
    elif n < 0:
        a = inv(a)
        n = abs(n)
    if n == 1:
        return a
    elif n == 2:
        return fmatmul(a, a)
    elif n == 3:
        return fmatmul(fmatmul(a, a), a)
    z = result = None
    while n > 0:
        z = a if z is None else fmatmul(z, z)
        n, bit = divmod(n, 2)
        if bit:
            result = z if result is None else fmatmul(result, z)
    return result
@array_function_dispatch(_unary_dispatcher)
def cholesky(a):
    extobj = get_linalg_error_extobj(_raise_linalgerror_nonposdef)
    gufunc = _umath_linalg.cholesky_lo
    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)
    signature = 'D->D' if isComplexType(t) else 'd->d'
    r = gufunc(a, signature=signature, extobj=extobj)
    return wrap(r.astype(result_t, copy=False))
def _qr_dispatcher(a, mode=None):
    return (a,)
@array_function_dispatch(_qr_dispatcher)
def qr(a, mode='reduced'):
    if mode not in ('reduced', 'complete', 'r', 'raw'):
        if mode in ('f', 'full'):
            msg = "".join((
                    "The 'full' option is deprecated in favor of 'reduced'.\n",
                    "For backward compatibility let mode default."))
            warnings.warn(msg, DeprecationWarning, stacklevel=3)
            mode = 'reduced'
        elif mode in ('e', 'economic'):
            msg = "The 'economic' option is deprecated."
            warnings.warn(msg, DeprecationWarning, stacklevel=3)
            mode = 'economic'
        else:
            raise ValueError(f"Unrecognized mode '{mode}'")
    a, wrap = _makearray(a)
    _assert_2d(a)
    m, n = a.shape
    t, result_t = _commonType(a)
    a = _fastCopyAndTranspose(t, a)
    a = _to_native_byte_order(a)
    mn = min(m, n)
    tau = zeros((mn,), t)
    if isComplexType(t):
        lapack_routine = lapack_lite.zgeqrf
        routine_name = 'zgeqrf'
    else:
        lapack_routine = lapack_lite.dgeqrf
        routine_name = 'dgeqrf'
    lwork = 1
    work = zeros((lwork,), t)
    results = lapack_routine(m, n, a, max(1, m), tau, work, -1, 0)
    if results['info'] != 0:
        raise LinAlgError('%s returns %d' % (routine_name, results['info']))
    lwork = max(1, n, int(abs(work[0])))
    work = zeros((lwork,), t)
    results = lapack_routine(m, n, a, max(1, m), tau, work, lwork, 0)
    if results['info'] != 0:
        raise LinAlgError('%s returns %d' % (routine_name, results['info']))
    if mode == 'r':
        r = _fastCopyAndTranspose(result_t, a[:, :mn])
        return wrap(triu(r))
    if mode == 'raw':
        return a, tau
    if mode == 'economic':
        if t != result_t :
            a = a.astype(result_t, copy=False)
        return wrap(a.T)
    if mode == 'complete' and m > n:
        mc = m
        q = empty((m, m), t)
    else:
        mc = mn
        q = empty((n, m), t)
    q[:n] = a
    if isComplexType(t):
        lapack_routine = lapack_lite.zungqr
        routine_name = 'zungqr'
    else:
        lapack_routine = lapack_lite.dorgqr
        routine_name = 'dorgqr'
    lwork = 1
    work = zeros((lwork,), t)
    results = lapack_routine(m, mc, mn, q, max(1, m), tau, work, -1, 0)
    if results['info'] != 0:
        raise LinAlgError('%s returns %d' % (routine_name, results['info']))
    lwork = max(1, n, int(abs(work[0])))
    work = zeros((lwork,), t)
    results = lapack_routine(m, mc, mn, q, max(1, m), tau, work, lwork, 0)
    if results['info'] != 0:
        raise LinAlgError('%s returns %d' % (routine_name, results['info']))
    q = _fastCopyAndTranspose(result_t, q[:mc])
    r = _fastCopyAndTranspose(result_t, a[:, :mc])
    return wrap(q), wrap(triu(r))
@array_function_dispatch(_unary_dispatcher)
def eigvals(a):
    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    _assert_finite(a)
    t, result_t = _commonType(a)
    extobj = get_linalg_error_extobj(
        _raise_linalgerror_eigenvalues_nonconvergence)
    signature = 'D->D' if isComplexType(t) else 'd->D'
    w = _umath_linalg.eigvals(a, signature=signature, extobj=extobj)
    if not isComplexType(t):
        if all(w.imag == 0):
            w = w.real
            result_t = _realType(result_t)
        else:
            result_t = _complexType(result_t)
    return w.astype(result_t, copy=False)
def _eigvalsh_dispatcher(a, UPLO=None):
    return (a,)
@array_function_dispatch(_eigvalsh_dispatcher)
def eigvalsh(a, UPLO='L'):
    UPLO = UPLO.upper()
    if UPLO not in ('L', 'U'):
        raise ValueError("UPLO argument must be 'L' or 'U'")
    extobj = get_linalg_error_extobj(
        _raise_linalgerror_eigenvalues_nonconvergence)
    if UPLO == 'L':
        gufunc = _umath_linalg.eigvalsh_lo
    else:
        gufunc = _umath_linalg.eigvalsh_up
    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)
    signature = 'D->d' if isComplexType(t) else 'd->d'
    w = gufunc(a, signature=signature, extobj=extobj)
    return w.astype(_realType(result_t), copy=False)
def _convertarray(a):
    t, result_t = _commonType(a)
    a = _fastCT(a.astype(t))
    return a, t, result_t
@array_function_dispatch(_unary_dispatcher)
def eig(a):
    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    _assert_finite(a)
    t, result_t = _commonType(a)
    extobj = get_linalg_error_extobj(
        _raise_linalgerror_eigenvalues_nonconvergence)
    signature = 'D->DD' if isComplexType(t) else 'd->DD'
    w, vt = _umath_linalg.eig(a, signature=signature, extobj=extobj)
    if not isComplexType(t) and all(w.imag == 0.0):
        w = w.real
        vt = vt.real
        result_t = _realType(result_t)
    else:
        result_t = _complexType(result_t)
    vt = vt.astype(result_t, copy=False)
    return w.astype(result_t, copy=False), wrap(vt)
@array_function_dispatch(_eigvalsh_dispatcher)
def eigh(a, UPLO='L'):
    UPLO = UPLO.upper()
    if UPLO not in ('L', 'U'):
        raise ValueError("UPLO argument must be 'L' or 'U'")
    a, wrap = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)
    extobj = get_linalg_error_extobj(
        _raise_linalgerror_eigenvalues_nonconvergence)
    if UPLO == 'L':
        gufunc = _umath_linalg.eigh_lo
    else:
        gufunc = _umath_linalg.eigh_up
    signature = 'D->dD' if isComplexType(t) else 'd->dd'
    w, vt = gufunc(a, signature=signature, extobj=extobj)
    w = w.astype(_realType(result_t), copy=False)
    vt = vt.astype(result_t, copy=False)
    return w, wrap(vt)
def _svd_dispatcher(a, full_matrices=None, compute_uv=None, hermitian=None):
    return (a,)
@array_function_dispatch(_svd_dispatcher)
def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
    import numpy as _nx
    a, wrap = _makearray(a)
    if hermitian:
        if compute_uv:
            s, u = eigh(a)
            sgn = sign(s)
            s = abs(s)
            sidx = argsort(s)[..., ::-1]
            sgn = _nx.take_along_axis(sgn, sidx, axis=-1)
            s = _nx.take_along_axis(s, sidx, axis=-1)
            u = _nx.take_along_axis(u, sidx[..., None, :], axis=-1)
            vt = transpose(u * sgn[..., None, :]).conjugate()
            return wrap(u), s, wrap(vt)
        else:
            s = eigvalsh(a)
            s = s[..., ::-1]
            s = abs(s)
            return sort(s)[..., ::-1]
    _assert_stacked_2d(a)
    t, result_t = _commonType(a)
    extobj = get_linalg_error_extobj(_raise_linalgerror_svd_nonconvergence)
    m, n = a.shape[-2:]
    if compute_uv:
        if full_matrices:
            if m < n:
                gufunc = _umath_linalg.svd_m_f
            else:
                gufunc = _umath_linalg.svd_n_f
        else:
            if m < n:
                gufunc = _umath_linalg.svd_m_s
            else:
                gufunc = _umath_linalg.svd_n_s
        signature = 'D->DdD' if isComplexType(t) else 'd->ddd'
        u, s, vh = gufunc(a, signature=signature, extobj=extobj)
        u = u.astype(result_t, copy=False)
        s = s.astype(_realType(result_t), copy=False)
        vh = vh.astype(result_t, copy=False)
        return wrap(u), s, wrap(vh)
    else:
        if m < n:
            gufunc = _umath_linalg.svd_m
        else:
            gufunc = _umath_linalg.svd_n
        signature = 'D->d' if isComplexType(t) else 'd->d'
        s = gufunc(a, signature=signature, extobj=extobj)
        s = s.astype(_realType(result_t), copy=False)
        return s
def _cond_dispatcher(x, p=None):
    return (x,)
@array_function_dispatch(_cond_dispatcher)
def cond(x, p=None):
    x = asarray(x)
    if _is_empty_2d(x):
        raise LinAlgError("cond is not defined on empty arrays")
    if p is None or p == 2 or p == -2:
        s = svd(x, compute_uv=False)
        with errstate(all='ignore'):
            if p == -2:
                r = s[..., -1] / s[..., 0]
            else:
                r = s[..., 0] / s[..., -1]
    else:
        _assert_stacked_2d(x)
        _assert_stacked_square(x)
        t, result_t = _commonType(x)
        signature = 'D->D' if isComplexType(t) else 'd->d'
        with errstate(all='ignore'):
            invx = _umath_linalg.inv(x, signature=signature)
            r = norm(x, p, axis=(-2, -1)) * norm(invx, p, axis=(-2, -1))
        r = r.astype(result_t, copy=False)
    r = asarray(r)
    nan_mask = isnan(r)
    if nan_mask.any():
        nan_mask &= ~isnan(x).any(axis=(-2, -1))
        if r.ndim > 0:
            r[nan_mask] = Inf
        elif nan_mask:
            r[()] = Inf
    if r.ndim == 0:
        r = r[()]
    return r
def _matrix_rank_dispatcher(M, tol=None, hermitian=None):
    return (M,)
@array_function_dispatch(_matrix_rank_dispatcher)
def matrix_rank(M, tol=None, hermitian=False):
    M = asarray(M)
    if M.ndim < 2:
        return int(not all(M==0))
    S = svd(M, compute_uv=False, hermitian=hermitian)
    if tol is None:
        tol = S.max(axis=-1, keepdims=True) * max(M.shape[-2:]) * finfo(S.dtype).eps
    else:
        tol = asarray(tol)[..., newaxis]
    return count_nonzero(S > tol, axis=-1)
def _pinv_dispatcher(a, rcond=None, hermitian=None):
    return (a,)
@array_function_dispatch(_pinv_dispatcher)
def pinv(a, rcond=1e-15, hermitian=False):
    a, wrap = _makearray(a)
    rcond = asarray(rcond)
    if _is_empty_2d(a):
        m, n = a.shape[-2:]
        res = empty(a.shape[:-2] + (n, m), dtype=a.dtype)
        return wrap(res)
    a = a.conjugate()
    u, s, vt = svd(a, full_matrices=False, hermitian=hermitian)
    cutoff = rcond[..., newaxis] * amax(s, axis=-1, keepdims=True)
    large = s > cutoff
    s = divide(1, s, where=large, out=s)
    s[~large] = 0
    res = matmul(transpose(vt), multiply(s[..., newaxis], transpose(u)))
    return wrap(res)
@array_function_dispatch(_unary_dispatcher)
def slogdet(a):
    a = asarray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)
    real_t = _realType(result_t)
    signature = 'D->Dd' if isComplexType(t) else 'd->dd'
    sign, logdet = _umath_linalg.slogdet(a, signature=signature)
    sign = sign.astype(result_t, copy=False)
    logdet = logdet.astype(real_t, copy=False)
    return sign, logdet
@array_function_dispatch(_unary_dispatcher)
def det(a):
    a = asarray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    t, result_t = _commonType(a)
    signature = 'D->D' if isComplexType(t) else 'd->d'
    r = _umath_linalg.det(a, signature=signature)
    r = r.astype(result_t, copy=False)
    return r
def _lstsq_dispatcher(a, b, rcond=None):
    return (a, b)
@array_function_dispatch(_lstsq_dispatcher)
def lstsq(a, b, rcond="warn"):
    a, _ = _makearray(a)
    b, wrap = _makearray(b)
    is_1d = b.ndim == 1
    if is_1d:
        b = b[:, newaxis]
    _assert_2d(a, b)
    m, n = a.shape[-2:]
    m2, n_rhs = b.shape[-2:]
    if m != m2:
        raise LinAlgError('Incompatible dimensions')
    t, result_t = _commonType(a, b)
    real_t = _linalgRealType(t)
    result_real_t = _realType(result_t)
    if rcond == "warn":
        warnings.warn("`rcond` parameter will change to the default of "
                      "machine precision times ``max(M, N)`` where M and N "
                      "are the input matrix dimensions.\n"
                      "To use the future default and silence this warning "
                      "we advise to pass `rcond=None`, to keep using the old, "
                      "explicitly pass `rcond=-1`.",
                      FutureWarning, stacklevel=3)
        rcond = -1
    if rcond is None:
        rcond = finfo(t).eps * max(n, m)
    if m <= n:
        gufunc = _umath_linalg.lstsq_m
    else:
        gufunc = _umath_linalg.lstsq_n
    signature = 'DDd->Ddid' if isComplexType(t) else 'ddd->ddid'
    extobj = get_linalg_error_extobj(_raise_linalgerror_lstsq)
    if n_rhs == 0:
        b = zeros(b.shape[:-2] + (m, n_rhs + 1), dtype=b.dtype)
    x, resids, rank, s = gufunc(a, b, rcond, signature=signature, extobj=extobj)
    if m == 0:
        x[...] = 0
    if n_rhs == 0:
        x = x[..., :n_rhs]
        resids = resids[..., :n_rhs]
    if is_1d:
        x = x.squeeze(axis=-1)
    if rank != n or m <= n:
        resids = array([], result_real_t)
    s = s.astype(result_real_t, copy=False)
    resids = resids.astype(result_real_t, copy=False)
    x = x.astype(result_t, copy=True)
    return wrap(x), wrap(resids), rank, s
def _multi_svd_norm(x, row_axis, col_axis, op):
    y = moveaxis(x, (row_axis, col_axis), (-2, -1))
    result = op(svd(y, compute_uv=False), axis=-1)
    return result
def _norm_dispatcher(x, ord=None, axis=None, keepdims=None):
    return (x,)
@array_function_dispatch(_norm_dispatcher)
def norm(x, ord=None, axis=None, keepdims=False):
    x = asarray(x)
    if not issubclass(x.dtype.type, (inexact, object_)):
        x = x.astype(float)
    if axis is None:
        ndim = x.ndim
        if ((ord is None) or
            (ord in ('f', 'fro') and ndim == 2) or
            (ord == 2 and ndim == 1)):
            x = x.ravel(order='K')
            if isComplexType(x.dtype.type):
                sqnorm = dot(x.real, x.real) + dot(x.imag, x.imag)
            else:
                sqnorm = dot(x, x)
            ret = sqrt(sqnorm)
            if keepdims:
                ret = ret.reshape(ndim*[1])
            return ret
    nd = x.ndim
    if axis is None:
        axis = tuple(range(nd))
    elif not isinstance(axis, tuple):
        try:
            axis = int(axis)
        except Exception as e:
            raise TypeError("'axis' must be None, an integer or a tuple of integers") from e
        axis = (axis,)
    if len(axis) == 1:
        if ord == Inf:
            return abs(x).max(axis=axis, keepdims=keepdims)
        elif ord == -Inf:
            return abs(x).min(axis=axis, keepdims=keepdims)
        elif ord == 0:
            return (x != 0).astype(x.real.dtype).sum(axis=axis, keepdims=keepdims)
        elif ord == 1:
            return add.reduce(abs(x), axis=axis, keepdims=keepdims)
        elif ord is None or ord == 2:
            s = (x.conj() * x).real
            return sqrt(add.reduce(s, axis=axis, keepdims=keepdims))
        elif isinstance(ord, str):
            raise ValueError(f"Invalid norm order '{ord}' for vectors")
        else:
            absx = abs(x)
            absx **= ord
            ret = add.reduce(absx, axis=axis, keepdims=keepdims)
            ret **= (1 / ord)
            return ret
    elif len(axis) == 2:
        row_axis, col_axis = axis
        row_axis = normalize_axis_index(row_axis, nd)
        col_axis = normalize_axis_index(col_axis, nd)
        if row_axis == col_axis:
            raise ValueError('Duplicate axes given.')
        if ord == 2:
            ret =  _multi_svd_norm(x, row_axis, col_axis, amax)
        elif ord == -2:
            ret = _multi_svd_norm(x, row_axis, col_axis, amin)
        elif ord == 1:
            if col_axis > row_axis:
                col_axis -= 1
            ret = add.reduce(abs(x), axis=row_axis).max(axis=col_axis)
        elif ord == Inf:
            if row_axis > col_axis:
                row_axis -= 1
            ret = add.reduce(abs(x), axis=col_axis).max(axis=row_axis)
        elif ord == -1:
            if col_axis > row_axis:
                col_axis -= 1
            ret = add.reduce(abs(x), axis=row_axis).min(axis=col_axis)
        elif ord == -Inf:
            if row_axis > col_axis:
                row_axis -= 1
            ret = add.reduce(abs(x), axis=col_axis).min(axis=row_axis)
        elif ord in [None, 'fro', 'f']:
            ret = sqrt(add.reduce((x.conj() * x).real, axis=axis))
        elif ord == 'nuc':
            ret = _multi_svd_norm(x, row_axis, col_axis, sum)
        else:
            raise ValueError("Invalid norm order for matrices.")
        if keepdims:
            ret_shape = list(x.shape)
            ret_shape[axis[0]] = 1
            ret_shape[axis[1]] = 1
            ret = ret.reshape(ret_shape)
        return ret
    else:
        raise ValueError("Improper number of dimensions to norm.")
def _multidot_dispatcher(arrays, *, out=None):
    yield from arrays
    yield out
@array_function_dispatch(_multidot_dispatcher)
def multi_dot(arrays, *, out=None):
    n = len(arrays)
    if n < 2:
        raise ValueError("Expecting at least two arrays.")
    elif n == 2:
        return dot(arrays[0], arrays[1], out=out)
    arrays = [asanyarray(a) for a in arrays]
    ndim_first, ndim_last = arrays[0].ndim, arrays[-1].ndim
    if arrays[0].ndim == 1:
        arrays[0] = atleast_2d(arrays[0])
    if arrays[-1].ndim == 1:
        arrays[-1] = atleast_2d(arrays[-1]).T
    _assert_2d(*arrays)
    if n == 3:
        result = _multi_dot_three(arrays[0], arrays[1], arrays[2], out=out)
    else:
        order = _multi_dot_matrix_chain_order(arrays)
        result = _multi_dot(arrays, order, 0, n - 1, out=out)
    if ndim_first == 1 and ndim_last == 1:
        return result[0, 0]
    elif ndim_first == 1 or ndim_last == 1:
        return result.ravel()
    else:
        return result
def _multi_dot_three(A, B, C, out=None):
    a0, a1b0 = A.shape
    b1c0, c1 = C.shape
    cost1 = a0 * b1c0 * (a1b0 + c1)
    cost2 = a1b0 * c1 * (a0 + b1c0)
    if cost1 < cost2:
        return dot(dot(A, B), C, out=out)
    else:
        return dot(A, dot(B, C), out=out)
def _multi_dot_matrix_chain_order(arrays, return_costs=False):
    n = len(arrays)
    p = [a.shape[0] for a in arrays] + [arrays[-1].shape[1]]
    m = zeros((n, n), dtype=double)
    s = empty((n, n), dtype=intp)
    for l in range(1, n):
        for i in range(n - l):
            j = i + l
            m[i, j] = Inf
            for k in range(i, j):
                q = m[i, k] + m[k+1, j] + p[i]*p[k+1]*p[j+1]
                if q < m[i, j]:
                    m[i, j] = q
                    s[i, j] = k
    return (s, m) if return_costs else s
def _multi_dot(arrays, order, i, j, out=None):
    if i == j:
        assert out is None
        return arrays[i]
    else:
        return dot(_multi_dot(arrays, order, i, order[i, j]),
                   _multi_dot(arrays, order, order[i, j] + 1, j),
                   out=out)
