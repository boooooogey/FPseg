import cffi
from numbers import Number
from _poisseg import ffi, lib
import numpy as np

def poisseg(y,lam,w=1,maxseglength=3000,averagerangelength=4):
    n = len(y)

    if isinstance(w,Number) and type(w) != bool:
        w = np.asarray(np.repeat(w,n),dtype=np.float64)

    if isinstance(lam,Number) and type(lam) != bool:
        lam = np.asarray(np.repeat(lam,n-1),dtype=np.float64)

    z = np.empty(n)

    maxseglength = int(maxseglength)
    averagerangelength = int(averagerangelength)

    assert len(w) == n
    assert len(lam) == n-1
    assert y.dtype == np.float64
    assert w.dtype == np.float64
    assert lam.dtype == np.float64
    assert z.dtype == np.float64
    assert type(maxseglength) == int
    assert type(averagerangelength) == int

    lib.L0PoisErrSeg(ffi.from_buffer('double[]',y),ffi.from_buffer('double[]',lam),ffi.from_buffer('double[]',w),ffi.cast("const int",n),ffi.from_buffer('double[]',z),ffi.cast('int', maxseglength),ffi.cast('int',averagerangelength))

    return np.exp(z)

def poissegbreakpoints(y,lam,w=1,maxseglength=3000,averagerangelength=4):
    n = len(y)

    if isinstance(w,Number) and type(w) != bool:
        w = np.asarray(np.repeat(w,n),dtype=np.float64)

    if isinstance(lam,Number) and type(lam) != bool:
        lam = np.asarray(np.repeat(lam,n-1),dtype=np.float64)

    ii = np.empty(n,dtype=np.uint64)
    vals = np.empty(n,dtype=np.float64)
    k = np.empty(1,dtype=np.uint64)

    maxseglength = int(maxseglength)
    averagerangelength = int(averagerangelength)

    assert len(w) == n
    assert len(lam) == n-1
    assert len(ii) == n
    assert len(k) == 1
    assert y.dtype == np.float64
    assert w.dtype == np.float64
    assert lam.dtype == np.float64
    assert vals.dtype == np.float64
    assert ii.dtype == np.uint64
    assert k.dtype == np.uint64

    lib.L0PoisBreakPoints(ffi.from_buffer('double[]',y),ffi.from_buffer('double[]',lam),ffi.from_buffer('double[]',w),ffi.cast('const int',n),ffi.from_buffer('double[]',vals),ffi.from_buffer('uint64_t[]', ii), ffi.from_buffer('uint64_t[]', k),ffi.cast('int', maxseglength),ffi.cast('int',averagerangelength))
    k = int(k[0])

    return (np.hstack([np.array([0],dtype=np.uint64),ii[:k-1]]), ii[:k], np.exp(vals[:k]))
