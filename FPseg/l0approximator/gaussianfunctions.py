from numbers import Number
from _gaussianfunctions import ffi, lib
import numpy as np

def _checkinputs(y, lam, w):
    n = len(y)

    assert y.dtype == np.float64

    if isinstance(w,Number) and type(w) != bool:
        w = np.asarray(np.repeat(w,n),dtype=np.float64)
    else:
        assert w.dtype == np.float64
        assert len(w) == n

    if isinstance(lam,Number) and type(lam) != bool:
        lam = np.asarray(np.repeat(lam,n-1),dtype=np.float64)
    else:
        assert lam.dtype == np.float64
        assert len(lam) == n-1

    x = np.empty(n, dtype=np.float64)
    return lam, w, x, n

def _checkinputsCondensed(y, lam, w):
    n = len(y)

    assert y.dtype == np.float64

    if isinstance(w,Number) and type(w) != bool:
        w = np.asarray(np.repeat(w,n),dtype=np.float64)
    else:
        assert w.dtype == np.float64
        assert len(w) == n

    if isinstance(lam,Number) and type(lam) != bool:
        lam = np.asarray(np.repeat(lam,n-1),dtype=np.float64)
    else:
        assert lam.dtype == np.float64
        assert len(lam) == n-1

    s = np.empty(n, dtype=np.int32)
    e = np.empty(n, dtype=np.int32)
    v = np.empty(n, dtype=np.float64)
    return lam, w, s, e, v, n

def _checkinputsN(y, w):
    n = len(y)

    assert y.dtype == np.float64

    if isinstance(w,Number) and type(w) != bool:
        w = np.asarray(np.repeat(w,n),dtype=np.float64)
    else:
        assert w.dtype == np.float64
        assert len(w) == n

    x = np.empty(n, dtype=np.float64)
    return w, x, n

def _checkinputsNCondensed(y, N, w):
    n = len(y)

    assert y.dtype == np.float64

    if isinstance(w,Number) and type(w) != bool:
        w = np.asarray(np.repeat(w,n),dtype=np.float64)
    else:
        assert w.dtype == np.float64
        assert len(w) == n

    s = np.empty(N, dtype=np.int32)
    e = np.empty(N, dtype=np.int32)
    v = np.empty(N, dtype=np.float64)
    return w, s, e, v, n

def l0gaussianapproximate(y, lam, w=1):

    lam, w, x, n = _checkinputs(y, lam, w)

    lib.L0GaussianApproximate(ffi.cast('const int', int(n)), ffi.from_buffer('double[]', y), ffi.from_buffer('double[]', lam), ffi.from_buffer('double[]' ,w), ffi.from_buffer('double[]', x))

    return x

def l0gaussianapproximateCondensed(y, lam, w=1):

    lam, w, s, e, v, n = _checkinputsCondensed(y, lam, w)

    k = lib.L0GaussianApproximateCondensed(ffi.cast('const int', n), ffi.from_buffer('double[]', y), ffi.from_buffer('double[]', lam), ffi.from_buffer('double[]' ,w), ffi.from_buffer('int[]', s), ffi.from_buffer('int[]', e), ffi.from_buffer('double[]', v))
    k -= 1

    return s[k::-1], e[k::-1], v[k::-1]

def l0gaussianapproximateN(y, N, w=1):

    w, x, n = _checkinputsN(y, w)

    lib.L0GaussianApproximateN(ffi.cast('const int', int(n)), ffi.from_buffer('double[]', y), ffi.cast('const int', N), ffi.from_buffer('double[]' ,w), ffi.from_buffer('double[]', x))

    return x

def l0gaussianapproximateNCondensed(y, N, w=1):

    w, s, e, v, n = _checkinputsNCondensed(y, N, w)

    k = lib.L0GaussianApproximateNCondensed(ffi.cast('const int', n), ffi.from_buffer('double[]', y), ffi.cast('const int', N), ffi.from_buffer('double[]' ,w), ffi.from_buffer('int[]', s), ffi.from_buffer('int[]', e), ffi.from_buffer('double[]', v))

    assert k == N

    return s[::-1], e[::-1], v[::-1]

def l0gaussianbreakpoint(y, w=1):
    w, _, _, _, n = _checkinputsNCondensed(y, 1, w)
    k = lib.L0GaussianBreakPoint(ffi.cast('const int', n), ffi.from_buffer('double[]', y), ffi.from_buffer('double[]' ,w))
    return k

