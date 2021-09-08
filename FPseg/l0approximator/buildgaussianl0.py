import cffi

ffi = cffi.FFI()

ffi.cdef("void L0GaussianApproximate(const int n, const double* y, const double* l, const double* w, double* x);")

ffi.cdef("int L0GaussianApproximateCondensed(const int n, const double* y, const double* l, const double* w, int* start, int* end, double* value);")

ffi.cdef("void L0GaussianApproximateN(const int n, const double* y, const int N, const double* w, double* x);")

ffi.cdef("int L0GaussianApproximateNCondensed(const int n, const double* y, const int N, const double* w, int* start, int* end, double* value);")

ffi.cdef("int L0GaussianBreakPoint(const int n, const double* y, const double* w);")

ffi.set_source("_gaussianfunctions",
        '''
        void L0GaussianApproximate(const int n, const double* y, const double* l, const double* w, double* x);
        int L0GaussianApproximateCondensed(const int n, const double* y, const double* l, const double* w, int* start, int* end, double* value);
        void L0GaussianApproximateN(const int n, const double* y, const int N, const double* w, double* x);
        int L0GaussianApproximateNCondensed(const int n, const double* y, const int N, const double* w, int* start, int* end, double* value);
        int L0GaussianBreakPoint(const int n, const double* y, const double* w);
        '''
        ,sources=["./FPseg/l0approximator/gaussianfunctions.cpp", "./FPseg/l0approximator/functions/squarederror.cpp", "./FPseg/l0approximator/core/range.cpp", "./FPseg/l0approximator/core/util.cpp"],
        source_extension=".cpp")

if __name__ == "__main__":
    ffi.compile(verbose=True)
