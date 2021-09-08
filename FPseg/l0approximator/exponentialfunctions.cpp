#include "functions/exponentialerror.hpp"
#include "core/util.hpp"

void L0ExponentialApproximate(const int n, const double* y, const double* l, const double* w, double* x){
    approximate<ExponentialError>(n, y, l, w, x);
}

int L0ExponentialApproximateCondensed(const int n, const double* y, const double* l, const double* w, int* start, int* end, double* value){
    int k;
    approximate<ExponentialError>(n, y, l, w, k, start, end, value);
    return k;
}

void L0ExponentialApproximateN(const int n, const double* y, const int N, const double* w, double* x){
    approximate<ExponentialError>(n, y, N, w, x);
}

int L0ExponentialApproximateNCondensed(const int n, const double* y, const int N, const double* w, int* start, int* end, double* value){
    int k;
    approximate<ExponentialError>(n, y, N, w, k, start, end, value);
    return k;
}

int L0ExponentialBreakPoint(const int n, const double* y, const double* w){
    return findbreakpoint<ExponentialError>(n, y, w);
}
