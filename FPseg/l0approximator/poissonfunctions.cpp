#include "functions/poissonerror.hpp"
#include "core/util.hpp"

void L0PoissonApproximate(const int n, const double* y, const double* l, const double* w, double* x){
    approximate<PoissonError>(n, y, l, w, x);
}

int L0PoissonApproximateCondensed(const int n, const double* y, const double* l, const double* w, int* start, int* end, double* value){
    int k;
    approximate<PoissonError>(n, y, l, w, k, start, end, value);
    return k;
}

void L0PoissonApproximateN(const int n, const double* y, const int N, const double* w, double* x){
    approximate<PoissonError>(n, y, N, w, x);
}

int L0PoissonApproximateNCondensed(const int n, const double* y, const int N, const double* w, int* start, int* end, double* value){
    int k;
    approximate<PoissonError>(n, y, N, w, k, start, end, value);
    return k;
}

int L0PoissonBreakPoint(const int n, const double* y, const double* w){
    return findbreakpoint<PoissonError>(n, y, w);
}
