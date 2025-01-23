#include "subtractor.cuh"
#include <utils/arithmetic.hpp>
#include <bigint/mixin/random.hpp>
#include <bigint/mixin/montgomery.hpp>
#include <curve/mixin/jacobian.hpp>
#include <utils.hpp>
#include <curve.hpp>
#include <bigint.hpp>

namespace gec
{
    namespace curve
    {
        namespace secp256k1
        {

            __global__ void setup_rand_kernel(curandState *states, unsigned long seed)
            {
                int idx = threadIdx.x + blockIdx.x * blockDim.x;
                curand_init(seed, idx, 0, &states[idx]);
            }

            __global__ void subtract_pubkey_kernel(
                curandState *states,
                SubtractionResult *results,
                const Curve<> *input,
                const Curve<> *d_gen,
                size_t batch_size,
                bool *success_flags)
            {
                const int idx = threadIdx.x + blockIdx.x * blockDim.x;
                if (idx >= batch_size)
                    return;

                curandState local_state = states[idx];

                // Random scalar generation (unchanged)
                uint64_t r1 = ((uint64_t)curand(&local_state) << 32) | curand(&local_state);
                uint64_t r2 = ((uint64_t)curand(&local_state) << 32) | curand(&local_state);
                uint64_t r3 = ((uint64_t)curand(&local_state) << 32) | curand(&local_state);

                results[idx].scalar.array()[3] = 0;
                results[idx].scalar.array()[2] = 0x40ULL + (r1 % 0x40ULL);
                results[idx].scalar.array()[1] = r2;
                results[idx].scalar.array()[0] = r3;

                // Convert scalar to Montgomery form
                Field::to_montgomery(results[idx].scalar, results[idx].scalar);

                Curve<> gen_mult;
                Curve<>::mul(gen_mult, results[idx].scalar, *d_gen);
                Field::neg(gen_mult.y(), gen_mult.y());
                Curve<>::add(results[idx].result, *input, gen_mult);
                Curve<>::to_affine(results[idx].result);

                states[idx] = local_state;
                success_flags[idx] = true;
            }

            GEC_H cudaError_t initialize_subtractor(
                SubtractorBatch &batch,
                const Curve<> &input_point,
                const Scalar &range_start,
                const Scalar &range_end)
            {
                cudaError_t err;

                // Convert input point to Montgomery form
                Curve<> mont_input = input_point;
                Field::to_montgomery(mont_input.x(), mont_input.x());
                Field::to_montgomery(mont_input.y(), mont_input.y());
                Field::to_montgomery(mont_input.z(), mont_input.z());

                err = cudaMemcpy(batch.input_point(), &mont_input, sizeof(Curve<>), cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                    return err;

                err = cudaMemcpy(batch.range_start(), &range_start, sizeof(Scalar), cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                    return err;

                err = cudaMemcpy(batch.range_end(), &range_end, sizeof(Scalar), cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                    return err;

                setup_rand_kernel<<<(batch.size() + 255) / 256, 256>>>(batch.states(), time(NULL));
                return cudaDeviceSynchronize();
            }

            GEC_H cudaError_t launch_subtraction(
                SubtractorBatch &batch,
                const dim3 grid,
                const dim3 block)
            {
                bool *d_success_flags = nullptr;
                Curve<> *d_gen = nullptr;
                cudaError_t err;

                err = cudaMalloc(&d_success_flags, batch.size() * sizeof(bool));
                if (err != cudaSuccess)
                    return err;

                err = cudaMalloc(&d_gen, sizeof(Curve<>));
                if (err != cudaSuccess)
                {
                    cudaFree(d_success_flags);
                    return err;
                }

                Curve<> mont_gen = Gen;
                Field::to_montgomery(mont_gen.x(), mont_gen.x());
                Field::to_montgomery(mont_gen.y(), mont_gen.y());
                Field::to_montgomery(mont_gen.z(), mont_gen.z());

                err = cudaMemcpy(d_gen, &mont_gen, sizeof(Curve<>), cudaMemcpyHostToDevice);
                if (err != cudaSuccess)
                {
                    cudaFree(d_success_flags);
                    cudaFree(d_gen);
                    return err;
                }

                subtract_pubkey_kernel<<<grid, block>>>(
                    batch.states(),
                    batch.results(),
                    batch.input_point(),
                    d_gen,
                    batch.size(),
                    d_success_flags);

                err = cudaGetLastError();

                cudaFree(d_success_flags);
                cudaFree(d_gen);

                return err;
            }

        }
    }
} // namespace gec::curve::secp256k1
