#include "tensor.hpp"
#include <iostream>
#include <cassert>

int main() {
    Tensor A({2, 2});
    Tensor B({2, 2});

    A[0] = 1; A[1] = 2; A[2] = 3; A[3] = 4;
    B[0] = 5; B[1] = 6; B[2] = 7; B[3] = 8;

    Tensor C = Tensor::matmul(A, B);
    
    std::cout << "C[0]: " << C[0] << " (expected 19)" << std::endl;
    std::cout << "C[1]: " << C[1] << " (expected 22)" << std::endl;
    std::cout << "C[2]: " << C[2] << " (expected 43)" << std::endl;
    std::cout << "C[3]: " << C[3] << " (expected 50)" << std::endl;

    assert(C[0] == 19 && C[1] == 22 && C[2] == 43 && C[3] == 50);
    std::cout << "Phase 4 matrix multiplication check passed!" << std::endl;
    return 0;
}
