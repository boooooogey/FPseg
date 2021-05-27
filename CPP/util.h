#ifndef _UTIL_H
#define _UTIL_H

extern double inf;
extern double neginf;
extern double infmin;

void backtrace(const double *, const double *, const int *, const int &, double *);
bool addRange(double *, const double &, int &, const int &);

#endif
