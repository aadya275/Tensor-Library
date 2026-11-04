#include "tensor.hpp"
#include <chrono>
#include <cstdio>

template <class F> double best_ms(F f, int reps) {
    double best = 1e18;
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        f();
        auto t1 = std::chrono::high_resolution_clock::now();
        best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return best;
}

int main() {
    printf("%6s  %12s  %12s  %9s\n", "N", "naive(ms)", "fast(ms)", "speedup");
    for (size_t n : {256, 512, 1024}) {
        Tensor A = Tensor::random({n, n}, 1), B = Tensor::random({n, n}, 2);
        double tn = best_ms([&]() { auto C = Tensor::matmul_naive(A, B); }, n == 512 ? 1 : 3);
        double tf = best_ms([&]() { auto C = Tensor::matmul(A, B); }, 5);
        printf("%6zu %12.2f %12.2f %8.1fx\n", n, tn, tf, tn / tf);
    }
    return 0;
}
