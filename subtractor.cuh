#pragma once

#include <curve/secp256k1.hpp>
#include <bigint/data/literal.hpp>
#include <bigint/data/array.hpp>
#include <bigint/mixin/division.hpp>
#include <bigint/mixin/add_sub.hpp>
#include <bigint/preset.hpp>
#include <cuda_runtime.h>
#ifdef __CUDACC__
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#endif
#include "bloom.h"
#include "xxhash.h"
#include <atomic>
#include <mutex>
#include <vector>
#include <chrono>
#include <random>

struct CustomRng
{
    unsigned long state;

#ifdef __CUDACC__
    __host__ __device__ CustomRng()
    {
#ifdef __CUDA_ARCH__
        state = clock64();
#else
        state = std::chrono::high_resolution_clock::now().time_since_epoch().count();
#endif
    }
#else
    CustomRng()
    {
        state = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }
#endif

    template <typename T>
#ifdef __CUDACC__
    __host__ __device__
#endif
        T
        operator()(T higher)
    {
        state = state * 6364136223846793005ULL + 1;
        return state % higher;
    }

    template <typename T>
#ifdef __CUDACC__
    __host__ __device__
#endif
        T
        operator()(T lower, T higher)
    {
        return lower + operator()(higher - lower);
    }

    template <typename T>
#ifdef __CUDACC__
    __host__ __device__
#endif
        T
        operator()()
    {
        state = state * 6364136223846793005ULL + 1;
        return static_cast<T>(state);
    }
};

namespace gec
{
    template <>
    struct GecRng<CustomRng>
    {
        CustomRng rng;

        GecRng(CustomRng r) : rng(r) {}

        template <typename T>
#ifdef __CUDACC__
        __host__ __device__
#endif
            T
            sample()
        {
            return rng.template operator()<T>();
        }

        template <typename T>
#ifdef __CUDACC__
        __host__ __device__
#endif
            T
            sample(T higher)
        {
            return rng(higher);
        }

        template <typename T>
#ifdef __CUDACC__
        __host__ __device__
#endif
            T
            sample(T lower, T higher)
        {
            return rng(lower, higher);
        }
    };
}

__global__ void subtract_kernel(
    const gec::curve::secp256k1::Curve<gec::curve::JacobianCurve> *input_point,
    const gec::curve::secp256k1::Scalar start,
    const gec::curve::secp256k1::Scalar step,
    const size_t batch_size,
    gec::curve::secp256k1::Curve<gec::curve::JacobianCurve> *results,
    gec::curve::secp256k1::Scalar *scalars);
extern const std::string RANGE_START;
extern const std::string RANGE_END;
gec::curve::secp256k1::Scalar parse_hex_string(const std::string &hex);

class PubkeySubtractor
{
public:
    // Constants
    static constexpr size_t MAX_ENTRIES1 = 10000000;
    static constexpr size_t MAX_ENTRIES2 = 8000000;
    static constexpr size_t MAX_ENTRIES3 = 6000000;
    static constexpr double BLOOM1_FP_RATE = 0.0001;
    static constexpr double BLOOM2_FP_RATE = 0.00001;
    static constexpr double BLOOM3_FP_RATE = 0.000001;
    static constexpr size_t PUBKEY_PREFIX_LENGTH = 6;

    using CurvePoint = gec::curve::secp256k1::Curve<gec::curve::JacobianCurve>;
    using Field = gec::curve::secp256k1::Field;
    using Scalar = gec::curve::secp256k1::Scalar;

    PubkeySubtractor(const std::string &pubkey, size_t batch_size);
    ~PubkeySubtractor();

    bool init_bloom_filters(const std::string &filename);
    void subtract_range(const Scalar &start, const Scalar &end, std::atomic<bool> &should_stop);
    gec::curve::secp256k1::Scalar parse_hex_string(const std::string &hex);

private:
    // Host data
    CurvePoint h_input_point;
    bloom h_bloom_filter1;
    bloom h_bloom_filter2;
    bloom h_bloom_filter3;
    std::atomic<uint64_t> attempts{0};
    std::chrono::steady_clock::time_point start_time;
    std::mutex cout_mutex;
    std::string generated_pubkey;
    std::string current_subtraction_value;
    size_t batch_size;
    bool bloom_initialized{false};

    // Device data
    CurvePoint *d_input_point{nullptr};
    CurvePoint *d_results{nullptr};
    Scalar *d_scalars{nullptr};

// CUDA streams and events
#ifdef __CUDACC__
    cudaStream_t compute_stream{nullptr};
    cudaStream_t transfer_stream{nullptr};
    cudaEvent_t compute_done{nullptr};
    cudaEvent_t transfer_done{nullptr};
#endif

    // Methods
    bool parse_pubkey(const std::string &pubkey);
    void report_status(bool final = false);
    void cleanup_device();
    void initialize_device();
    void save_match(const std::string &pubkey, const Scalar &scalar);
    bool check_bloom_filters(const std::string &compressed);
    void process_entries(const char *data, size_t num_entries, bool is_binary);
    void process_gpu_results(CurvePoint *points, Scalar *scalars, size_t valid_results);

    struct WorkerBuffer
    {
        std::vector<char> data;
        size_t used{0};
        WorkerBuffer() : data(1024 * 1024) {}
        void reset() { used = 0; }
    };
};
