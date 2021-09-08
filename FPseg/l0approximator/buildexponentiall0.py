import cffi

ffi = cffi.FFI()

ffi.cdef("void L0ExponentialApproximate(const int n, const double* y, const double* l, const double* w, double* x);")

ffi.cdef("int L0ExponentialApproximateCondensed(const int n, const double* y, const double* l, const double* w, int* start, int* end, double* value);")

ffi.cdef("void L0ExponentialApproximateN(const int n, const double* y, const int N, const double* w, double* x);")

ffi.cdef("int L0ExponentialApproximateNCondensed(const int n, const double* y, const int N, const double* w, int* start, int* end, double* value);")

ffi.cdef("int L0ExponentialBreakPoint(const int n, const double* y, const double* w);")

ffi.set_source("_exponentialfunctions",
        '''
        void L0ExponentialApproximate(const int n, const double* y, const double* l, const double* w, double* x);
        int L0ExponentialApproximateCondensed(const int n, const double* y, const double* l, const double* w, int* start, int* end, double* value);
        void L0ExponentialApproximateN(const int n, const double* y, const int N, const double* w, double* x);
        int L0ExponentialApproximateNCondensed(const int n, const double* y, const int N, const double* w, int* start, int* end, double* value);
        int L0ExponentialBreakPoint(const int n, const double* y, const double* w);
        '''
        ,sources=["./FPseg/l0approximator/exponentialfunctions.cpp", "./FPseg/l0approximator/functions/exponentialerror.cpp", "./FPseg/l0approximator/core/range.cpp", "./FPseg/l0approximator/core/util.cpp"],
        source_extension=".cpp")

if __name__ == "__main__":
    ffi.compile(verbose=True)
