#include "functions/squarederror.hpp"
#include "core/util.hpp"

void L0GaussianApproximate(const int n, const double* y, const double* l, const double* w, double* x){
    approximate<SquaredError>(n, y, l, w, x);
}

int L0GaussianApproximateCondensed(const int n, const double* y, const double* l, const double* w, int* start, int* end, double* value){
    int k;
    approximate<SquaredError>(n, y, l, w, k, start, end, value);
    return k;
}

void L0GaussianApproximateN(const int n, const double* y, const int N, const double* w, double* x){
    approximate<SquaredError>(n, y, N, w, x);
}

int L0GaussianApproximateNCondensed(const int n, const double* y, const int N, const double* w, int* start, int* end, double* value){
    int k;
    approximate<SquaredError>(n, y, N, w, k, start, end, value);
    return k;
}

int L0GaussianBreakPoint(const int n, const double* y, const double* w){
    return findbreakpoint<SquaredError>(n, y, w);
}
