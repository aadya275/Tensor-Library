#include "tensor.hpp"

int main() {
    Tensor X = Tensor::random({4, 3});
    Tensor W = Tensor::random({3, 2}, 7);
    Tensor Z = Tensor::matmul(X, W);
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 2; ++j) Z[i * 2 + j] += 0.1f;
    Tensor Y = Z.relu();
    Y.print();
    return 0;
}
