import cffi

ffi = cffi.FFI()

ffi.cdef("void L0PoissonApproximate(const int n, const double* y, const double* l, const double* w, double* x);")

ffi.cdef("int L0PoissonApproximateCondensed(const int n, const double* y, const double* l, const double* w, int* start, int* end, double* value);")

ffi.cdef("void L0PoissonApproximateN(const int n, const double* y, const int N, const double* w, double* x);")

ffi.cdef("int L0PoissonApproximateNCondensed(const int n, const double* y, const int N, const double* w, int* start, int* end, double* value);")

ffi.cdef("int L0PoissonBreakPoint(const int n, const double* y, const double* w);")

ffi.set_source("_poissonfunctions",
        '''
        void L0PoissonApproximate(const int n, const double* y, const double* l, const double* w, double* x);
        int L0PoissonApproximateCondensed(const int n, const double* y, const double* l, const double* w, int* start, int* end, double* value);
        void L0PoissonApproximateN(const int n, const double* y, const int N, const double* w, double* x);
        int L0PoissonApproximateNCondensed(const int n, const double* y, const int N, const double* w, int* start, int* end, double* value);
        int L0PoissonBreakPoint(const int n, const double* y, const double* w);
        '''
        ,sources=["./FPseg/l0approximator/poissonfunctions.cpp", "./FPseg/l0approximator/functions/poissonerror.cpp", "./FPseg/l0approximator/core/range.cpp", "./FPseg/l0approximator/core/util.cpp"],
        source_extension=".cpp")

if __name__ == "__main__":
    ffi.compile(verbose=True)
