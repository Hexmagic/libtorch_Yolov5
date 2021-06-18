#include "torch/script.h"
#include <iostream>
using namespace std;



int main(){
    auto mat=    torch::rand({2,3,4});
    auto rst = mat;
    cout<<rst.select(0,1)<<endl;
    cout<<rst.select(1,1)<<endl;
    cout<<rst.size(0)<<endl;
    cout<<rst.size(1)<<endl;
    cout<<rst.size(2)<<endl;
}