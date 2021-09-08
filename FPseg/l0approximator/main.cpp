#include "functions/squarederror.hpp"
#include "functions/poissonerror.hpp"
#include "functions/exponentialerror.hpp"
#include "core/piecewisefunction.hpp"
#include "core/range.hpp"
#include "core/util.hpp"
#include <fstream>
#include <vector>
#include <cstdlib>
#include <iostream>

int read_array(const char* filename, double*& array) {
    std::ifstream ifile(filename, std::ios::in);
    std::vector<double> scores;

    //check to see that the file was opened correctly:
    if (!ifile.is_open()) {
        std::cerr << "There was a problem opening the input file!\n";
        return 0;//exit or do additional error checking
    }

    double num = 0.0;
    //keep storing values from the text file so long as data exists:
    while (ifile >> num) {
        scores.push_back(num);
    }
    ifile.close();

    int n = scores.size();
    array = new double[n];

    std::copy(scores.begin(), scores.end(), array);
    //verify that the scores were stored correctly:
    //for (int i = 0; i < scores.size(); ++i) {
    //    std::cout << scores[i] << std::endl;
    //}
    return n;
}

template <typename T>
void write_array(const char* filename, T* array, const int& n){
    std::ofstream ofile(filename, std::ios::out);
    if (!ofile.is_open()) {
        std::cerr << "There was a problem opening the output file!\n";
        return;//exit or do additional error checking
    }
    for(int i = 0; i < n; i++){
        ofile << array[i] << std::endl;
    }
    ofile.close();
}

int main() {

    double* y;
    int n = read_array("/data/src/fps-package/FPseg/misc/experiments/experiment2.0/JunD.chr1.30M.40M.10.array",y);
    double lambda = 10;
    double* l = new double[n-1];
    for(int i = 0; i < n-1; i++){
        l[i] = lambda;
    }
    double* w = new double[n];
    for(int i = 0; i < n; i++){
        w[i] = 1;
    }
    double* x = new double[n];
    int k;
    int* start = new int[n];
    int* end = new int[n];
    double* val = new double[n];
    approximate<PoissonError>(n, y, l, w, k, start, end, val);
    write_array("start.array",start,k);
    write_array("end.array",end,k);
    write_array("val.array",val,k);
    int bp = findbreakpoint<PoissonError>(n, y, w);
    std::cout << bp << std::endl;

    delete[] y;
    delete[] l;
    delete[] w;
    delete[] start;
    delete[] end;
    delete[] val;
    delete[] x;

    //SquaredError f1(-1, -3, 1);
    //SquaredError f2(-1, 2, 5);
    //SquaredError f3(-1, 8, -15);
    //SquaredError f4(-SquaredError::rangeninf);

    //double x1, x2;
    //bool left, right;
    //f2.solve(2, x1, x2, left, right);
    //std::cout << x1 << " " << x2 << std::endl;

    //PiecewiseFunction<SquaredError> g;
    //g.append(f1, SquaredError::domainninf);
    //g.append(f2, -0.8);
    //g.append(f3, (double) 10/3);
    //g.append(f4, SquaredError::domaininf);

    //PiecewiseFunction<SquaredError> f;
    //RangeList r(1);
    //g.flood(2, f, r[0]);
    //std::cout << g.len() << std::endl;
    //std::cout << f.len() << std::endl;

    //std::cout << "g(-4) = " << g(-4) << std::endl;
    //std::cout << "f(-4) = " << f(-4) << std::endl;

    //std::cout << "g(-1) = " << g(-1) << std::endl;
    //std::cout << "f(-1) = " << f(-1) << std::endl;

    //std::cout << "g(2) = " << g(2) << std::endl;
    //std::cout << "f(2) = " << f(2) << std::endl;

    //std::cout << "g(4) = " << g(4) << std::endl;
    //std::cout << "f(4) = " << f(4) << std::endl;

    //double first, last;
    //for(int i = 0; i < r.len(0); i++){
    //    r.index(0, i, first, last);
    //    std::cout << "(" << first << "," << last << ")";
    //}
    //std::cout << std::endl;

    //SquaredError f5(2,1);
    //g = f + f5;

    //std::cout << "g(-4) = " << g(-4) << std::endl;
    //std::cout << "f(-4) = " << f(-4) << std::endl;

    //std::cout << "g(-1) = " << g(-1) << std::endl;
    //std::cout << "f(-1) = " << f(-1) << std::endl;

    //std::cout << "g(2) = " << g(2) << std::endl;
    //std::cout << "f(2) = " << f(2) << std::endl;

    //std::cout << "g(4) = " << g(4) << std::endl;
    //std::cout << "f(4) = " << f(4) << std::endl;

    return 0;

}
