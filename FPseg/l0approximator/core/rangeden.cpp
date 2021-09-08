#include "range.hpp"
#include <iostream>

int main(){
    RangeList a(3);
    a.add(0,1,1);
    a.add(0,3,1);
    a.add(0,5,1);
    a.add(1,2,2);
    a.add(1,2,4);
    a.add(1,2,6);
    a.add(2,1,3);
    a.add(2,3,3);
    a.add(2,5,3);
    double x, y;
    for (int i = 0; i < a.len(); i++){
        std::cout << "list " << i << ": ";
        for(int j = 0; j < a.len(i); j++){
            a.index(i,j,x,y);
            std::cout << "(" << x << "," << y << ") ";
        }
        std::cout << std::endl;
    }
    return 0;
}
