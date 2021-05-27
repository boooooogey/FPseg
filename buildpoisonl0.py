import cffi

ffi = cffi.FFI()

ffi.cdef("void L0PoisErrSeg(const double * y, const double * l2, const double * weights, const int N, double * z, int max_seg_length, int average_range_length);")

ffi.cdef("void L0PoisBreakPoints(const double * y, const double * l2, const double * weights, const int N, double * vals, uint64_t * ii, uint64_t * k, int max_seg_length, int average_range_length);")

ffi.set_source("_poisseg",
        '''
        void L0PoisErrSeg(const double * y, const double * l2, const double * weights, const int N, double * z, int max_seg_length, int average_range_length); 
        void L0PoisBreakPoints(const double * y, const double * l2, const double * weights, const int N, double * vals, uint64_t * ii, uint64_t * k, int max_seg_length, int average_range_length);
        '''
        ,sources=["CPP/poiserr.cpp","CPP/util.cpp"],source_extension=".cpp")

if __name__ == "__main__":
    ffi.compile(verbose=True)
