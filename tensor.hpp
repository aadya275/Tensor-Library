#pragma once
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <cmath>
#include <iostream>

class Tensor {
    std::vector<size_t> shape_, strides_;
    size_t size_;
    float* data_ = nullptr;

    void computeStrides() {
        strides_.assign(shape_.size(), 1);
        for (int i = (int)shape_.size() - 2; i >= 0; --i) {
            strides_[i] = strides_[i + 1] * shape_[i + 1];
        }
    }

    void allocate() {
        size_t bytes = ((size_ * sizeof(float) + 63) / 64) * 64;
        if (bytes == 0) bytes = 64;
        data_ = static_cast<float*>(std::aligned_alloc(64, bytes));
        if (!data_) throw std::bad_alloc();
    }

    void checkSame(const Tensor& o) const {
        if (shape_ != o.shape_) throw std::invalid_argument("shape mismatch");
    }

    static void checkMatmul(const Tensor& A, const Tensor& B) {
        if (A.shape_.size() != 2 || B.shape_.size() != 2 || A.shape_[1] != B.shape_[0])
            throw std::invalid_argument("matmul: need 2D tensors with A.cols == B.rows");
    }

    Tensor binary(const Tensor& o, int op) const {
        checkSame(o);
        Tensor r(shape_);
        const float* __restrict__ a = data_;
        const float* __restrict__ b = o.data_;
        float* __restrict__ c = r.data_;
        const size_t n = size_;

        if (op == 0) {
            #pragma omp parallel for simd if (n > 200000)
            for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
        } else if (op == 1) {
            #pragma omp parallel for simd if (n > 200000)
            for (size_t i = 0; i < n; ++i) c[i] = a[i] - b[i];
        } else {
            #pragma omp parallel for simd if (n > 200000)
            for (size_t i = 0; i < n; ++i) c[i] = a[i] * b[i];
        }
        return r;
    }

public:
    explicit Tensor(std::vector<size_t> shape) : shape_(std::move(shape)) {
        size_ = 1;
        for (auto s : shape_) size_ *= s;
        computeStrides();
        allocate();
        std::memset(data_, 0, size_ * sizeof(float));
    }

    ~Tensor() {
        std::free(data_);
    }

    Tensor(const Tensor& o) : shape_(o.shape_), strides_(o.strides_), size_(o.size_) {
        allocate();
        if (size_) std::memcpy(data_, o.data_, size_ * sizeof(float));
    }

    Tensor(Tensor&& o) noexcept
        : shape_(std::move(o.shape_)), strides_(std::move(o.strides_)),
          size_(o.size_), data_(o.data_) {
        o.data_ = nullptr;
        o.size_ = 0;
    }

    void swap(Tensor& o) noexcept {
        std::swap(shape_, o.shape_);
        std::swap(strides_, o.strides_);
        std::swap(size_, o.size_);
        std::swap(data_, o.data_);
    }

    Tensor& operator=(Tensor o) noexcept {
        swap(o);
        return *this;
    }

    // Static factories
    static Tensor zeros(std::vector<size_t> s) { return Tensor(std::move(s)); }
    static Tensor ones(std::vector<size_t> s) { Tensor t(std::move(s)); t.fill(1.0f); return t; }
    static Tensor random(std::vector<size_t> s, unsigned seed = 42) {
        Tensor t(std::move(s));
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> d(0.0f, 1.0f);
        for (size_t i = 0; i < t.size_; ++i) t.data_[i] = d(gen);
        return t;
    }

    void fill(float v) { for (size_t i = 0; i < size_; ++i) data_[i] = v; }

    // Getters & Indexing
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    size_t size() const { return size_; }
    size_t ndim() const { return shape_.size(); }
    float* data() { return data_; }
    const float* data() const { return data_; }

    float& operator[](size_t i) { return data_[i]; }
    const float& operator[](size_t i) const { return data_[i]; }

    // Math operations
    Tensor operator+(const Tensor& o) const { return binary(o, 0); }
    Tensor operator-(const Tensor& o) const { return binary(o, 1); }
    Tensor mul(const Tensor& o) const { return binary(o, 2); }

    Tensor operator*(float s) const {
        Tensor r(shape_);
        #pragma omp parallel for simd
        for (size_t i = 0; i < size_; ++i) r.data_[i] = data_[i] * s;
        return r;
    }

    Tensor operator+(float s) const {
        Tensor r(shape_);
        #pragma omp parallel for simd if(size_ > 200000)
        for (size_t i = 0; i < size_; ++i) r.data_[i] = data_[i] + s;
        return r;
    }

    void add_(const Tensor& o) {
        checkSame(o);
        #pragma omp parallel for simd if(size_ > 200000)
        for (size_t i = 0; i < size_; ++i) data_[i] += o.data_[i];
    }

    // Phase 4: Reshape, Transpose, Sum, ReLU
    Tensor reshape(std::vector<size_t> ns) const {
        size_t n = 1; for (auto s : ns) n *= s;
        if (n != size_) throw std::invalid_argument("reshape: total size must stay the same");
        Tensor t(*this);
        t.shape_ = std::move(ns);
        t.computeStrides();
        return t;
    }

    Tensor transpose() const {
        if (shape_.size() != 2) throw std::invalid_argument("transpose needs a 2D tensor");
        size_t R = shape_[0], C = shape_[1];
        Tensor t({C, R});
        const size_t B = 32;
        for (size_t ii = 0; ii < R; ii += B) {
            for (size_t jj = 0; jj < C; jj += B) {
                for (size_t i = ii; i < std::min(ii + B, R); ++i) {
                    for (size_t j = jj; j < std::min(jj + B, C); ++j) {
                        t.data_[j * R + i] = data_[i * C + j];
                    }
                }
            }
        }
        return t;
    }

    float sum() const {
        double s = 0;
        #pragma omp parallel for reduction(+:s) if(size_ > 200000)
        for (size_t i = 0; i < size_; ++i) s += data_[i];
        return (float)s;
    }

    Tensor relu() const {
        Tensor r(shape_);
        for (size_t i = 0; i < size_; ++i) r.data_[i] = data_[i] > 0 ? data_[i] : 0;
        return r;
    }

    static Tensor matmul_naive(const Tensor& A, const Tensor& B) {
        checkMatmul(A, B);
        size_t M = A.shape_[0], K = A.shape_[1], N = B.shape_[1];
        Tensor C({M, N});
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < N; ++j) {
                float s = 0;
                for (size_t k = 0; k < K; ++k) s += A.data_[i * K + k] * B.data_[k * N + j];
                C.data_[i * N + j] = s;
            }
        }
        return C;
    }

    static Tensor matmul(const Tensor& A, const Tensor& B) {
        checkMatmul(A, B);
        const size_t M = A.shape_[0], K = A.shape_[1], N = B.shape_[1];
        Tensor C({M, N});
        const float* __restrict__ a = A.data_;
        const float* __restrict__ b = B.data_;
        float* __restrict__ c = C.data_;
        const size_t BS = 64;

        #pragma omp parallel for schedule(static) if((double)M * N * K >= 1e6)
        for (size_t ii = 0; ii < M; ii += BS) {
            size_t iMax = std::min(ii + BS, M);
            for (size_t kk = 0; kk < K; kk += BS) {
                size_t kMax = std::min(kk + BS, K);
                for (size_t jj = 0; jj < N; jj += BS) {
                    size_t jMax = std::min(jj + BS, N);
                    for (size_t i = ii; i < iMax; ++i) {
                        for (size_t k = kk; k < kMax; ++k) {
                            float av = a[i * K + k];
                            for (size_t j = jj; j < jMax; ++j) {
                                c[i * N + j] += av * b[k * N + j];
                            }
                        }
                    }
                }
            }
        }
        return C;
    }

    void print() const {
        std::cout << "Tensor(shape=[";
        for (size_t i = 0; i < shape_.size(); ++i) std::cout << shape_[i] << (i + 1 < shape_.size() ? "," : "");
        std::cout << "]) data=[";
        for (size_t i = 0; i < std::min<size_t>(size_, 20); ++i) std::cout << data_[i] << " ";
        std::cout << (size_ > 20 ? "..." : "") << "]\n";
    }
};
