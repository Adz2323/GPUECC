#pragma once
#ifndef GEC_SUBTRACTOR_CUH
#define GEC_SUBTRACTOR_CUH

#include <curve/secp256k1.hpp>
#include <bigint/mixin.hpp>
#include <curve/mixin/scalar_mul.hpp>
#include <utils/cuda_utils.cuh>
#include <utils/basic.hpp>
#include <curand_kernel.h>
#include <utils.hpp>
#include <curve.hpp>
#include <bigint.hpp>

namespace gec
{
    namespace curve
    {
        namespace secp256k1
        {

            struct CustomRng
            {
                std::mt19937_64 gen;
                CustomRng() : gen(std::random_device{}()) {}

                template <typename T>
                GEC_HD T operator()(T higher) { return gen() % higher; }

                template <typename T>
                GEC_HD T operator()(T lower, T higher) { return lower + (gen() % (higher - lower)); }

                template <typename T>
                GEC_HD T operator()() { return static_cast<T>(gen()); }
            };

            struct alignas(16) SubtractionResult
            {
                using Point = Curve<>;
                Scalar scalar;
                Point result;

                GEC_HD void clear()
                {
                    scalar.set_zero();
                    result.set_inf();
                }
            };

            class SubtractorBatch
            {
            public:
                using Point = Curve<>;

                GEC_HD explicit SubtractorBatch(size_t batch_size) : size_(batch_size)
                {
                    cudaError_t err;
                    err = cudaMalloc(&d_states_, batch_size * sizeof(curandState));
                    if (err != cudaSuccess)
                    {
                        cleanup();
                        return;
                    }

                    err = cudaMalloc(&d_results_, batch_size * sizeof(SubtractionResult));
                    if (err != cudaSuccess)
                    {
                        cleanup();
                        return;
                    }

                    err = cudaMalloc(&d_input_point_, sizeof(Point));
                    if (err != cudaSuccess)
                    {
                        cleanup();
                        return;
                    }

                    err = cudaMalloc(&d_range_start_, sizeof(Scalar));
                    if (err != cudaSuccess)
                    {
                        cleanup();
                        return;
                    }

                    err = cudaMalloc(&d_range_end_, sizeof(Scalar));
                    if (err != cudaSuccess)
                    {
                        cleanup();
                        return;
                    }
                }

                GEC_HD ~SubtractorBatch()
                {
                    cleanup();
                }

                GEC_HD GEC_INLINE curandState *states() { return d_states_; }
                GEC_HD GEC_INLINE SubtractionResult *results() { return d_results_; }
                GEC_HD GEC_INLINE Point *input_point() { return d_input_point_; }
                GEC_HD GEC_INLINE Scalar *range_start() { return d_range_start_; }
                GEC_HD GEC_INLINE Scalar *range_end() { return d_range_end_; }
                GEC_HD GEC_INLINE size_t size() const { return size_; }

            private:
                void cleanup()
                {
                    if (d_states_)
                        cudaFree(d_states_);
                    if (d_results_)
                        cudaFree(d_results_);
                    if (d_input_point_)
                        cudaFree(d_input_point_);
                    if (d_range_start_)
                        cudaFree(d_range_start_);
                    if (d_range_end_)
                        cudaFree(d_range_end_);
                    d_states_ = nullptr;
                    d_results_ = nullptr;
                    d_input_point_ = nullptr;
                    d_range_start_ = nullptr;
                    d_range_end_ = nullptr;
                }

                size_t size_;
                curandState *d_states_ = nullptr;
                SubtractionResult *d_results_ = nullptr;
                Point *d_input_point_ = nullptr;
                Scalar *d_range_start_ = nullptr;
                Scalar *d_range_end_ = nullptr;
            };

            __global__ void setup_rand_kernel(curandState *states, unsigned long seed);

            __global__ void subtract_pubkey_kernel(
                curandState *states,
                SubtractionResult *results,
                const Curve<> *input,
                const Curve<> *d_gen,
                size_t batch_size,
                bool *success_flags);

            GEC_H cudaError_t initialize_subtractor(
                SubtractorBatch &batch,
                const Curve<> &input_point,
                const Scalar &range_start,
                const Scalar &range_end);

            GEC_H cudaError_t launch_subtraction(
                SubtractorBatch &batch,
                dim3 grid,
                dim3 block);

        }
    }
} // namespace gec::curve::secp256k1

#endif // GEC_SUBTRACTOR_CUH
