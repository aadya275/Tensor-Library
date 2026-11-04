#include "tensor.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

static bool close(float a, float b) { return std::fabs(a - b) < 1e-3f; }

int main() {
    // creation & indexing
    Tensor t(std::vector<size_t>({2, 3}));
    t[5] = 7;
    assert(t[5] == 7);
    assert(t.strides()[0] == 3 && t.strides()[1] == 1);

    // add/sub/mul
    Tensor a = Tensor::ones(std::vector<size_t>({2, 2}));
    Tensor b = Tensor::ones(std::vector<size_t>({2, 2})) * 3.0f;
    assert((a + b)[0] == 4 && (b - a)[0] == 2 && a.mul(b)[0] == 3);

    // matmul known answer
    Tensor A({2, 2}), B({2, 2});
    A[0] = 1; A[1] = 2; A[2] = 3; A[3] = 4;
    B[0] = 5; B[1] = 6; B[2] = 7; B[3] = 8;
    Tensor C = Tensor::matmul(A, B);
    assert(C[0] == 19 && C[1] == 22 && C[2] == 43 && C[3] == 50);

    // fast matmul matches naive on odd sizes
    Tensor X = Tensor::random({130, 77}, 1);
    Tensor Y = Tensor::random({77, 91}, 2);
    Tensor F = Tensor::matmul(X, Y);
    Tensor S = Tensor::matmul_naive(X, Y);
    for (size_t i = 0; i < F.size(); ++i) assert(close(F[i], S[i]));

    // reshape & transpose
    Tensor R = Tensor::random({4, 6}).reshape({2, 3, 4});
    assert(R.ndim() == 3 && R.size() == 24);
    Tensor T = A.transpose();
    assert(T[1] == 3 && T[2] == 2);

    // errors
    bool threw = false;
    try { a.reshape({3, 3}); } catch (const std::invalid_argument&) { threw = true; }
    assert(threw);

    std::cout << "ALL TESTS PASSED\n";
    return 0;
}
