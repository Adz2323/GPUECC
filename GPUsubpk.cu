#include <gec/curve/secp256k1.hpp>
#include <gec/bigint/mixin/random.hpp>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// GPU kernel for point subtraction
__global__ void point_subtraction_kernel(
    gec::curve::secp256k1::Point *points,
    const gec::curve::secp256k1::Point &input_point,
    const uint64_t *random_values,
    size_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    using Point = gec::curve::secp256k1::Curve<>;
    using Scalar = gec::curve::secp256k1::Scalar;

    // Create scalar from random value
    Scalar s;
    std::fill(s.array(), s.array() + 4, 0);
    s.array()[0] = random_values[idx];

    // Calculate generator multiplication
    Point gen_mult;
    Point::mul(gen_mult, s, gec::curve::secp256k1::d_Gen);

    // Negate y coordinate
    Field::neg(gen_mult.y(), gen_mult.y());

    // Add points
    Point result;
    Point::add(result, input_point, gen_mult);
    Point::to_affine(result);

    // Store result
    points[idx] = result;
}

class GPUPubkeySubtractor
{
    using Point = gec::curve::secp256k1::Curve<>;
    using Field = gec::curve::secp256k1::Field;
    using Scalar = gec::curve::secp256k1::Scalar;

    Point input_point;
    thrust::device_vector<Point> d_points;
    thrust::device_vector<uint64_t> d_random_values;
    thrust::host_vector<Point> h_points;

    // CUDA parameters
    static constexpr int BLOCK_SIZE = 256;
    int num_blocks;

public:
    GPUPubkeySubtractor(const std::string &pubkey, size_t batch_size)
        : d_points(batch_size), d_random_values(batch_size), h_points(batch_size)
    {
        parse_pubkey(pubkey);
        num_blocks = (batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    }

    void subtract_range(uint64_t start, uint64_t end)
    {
        const size_t batch_size = d_points.size();

        // Generate random values
        thrust::host_vector<uint64_t> h_random_values(batch_size);
        std::mt19937_64 rng(std::random_device{}());
        std::uniform_int_distribution<uint64_t> dist(start, end);
        for (size_t i = 0; i < batch_size; ++i)
        {
            h_random_values[i] = dist(rng);
        }

        // Copy random values to GPU
        d_random_values = h_random_values;

        // Launch kernel
        point_subtraction_kernel<<<num_blocks, BLOCK_SIZE>>>(
            thrust::raw_pointer_cast(d_points.data()),
            input_point,
            thrust::raw_pointer_cast(d_random_values.data()),
            batch_size);

        // Copy results back to host
        h_points = d_points;

        // Process results
        for (size_t i = 0; i < batch_size; ++i)
        {
            const auto &result = h_points[i];
            std::string compressed = to_compressed_hex(result.x(), result.y());
            std::cout << compressed << " (subtracted " << h_random_values[i] << ")\n";
        }
    }

private:
    void parse_pubkey(const std::string &pubkey)
    {
        // Same pubkey parsing logic as before
        if (pubkey.length() != 66 || pubkey[0] != '0' ||
            (pubkey[1] != '2' && pubkey[1] != '3'))
        {
            throw std::runtime_error("Invalid compressed public key format");
        }

        bool is_odd = (pubkey[1] == '3');
        std::string x_str = pubkey.substr(2);
        Field x;

        for (int i = 0; i < 4; i++)
        {
            std::string chunk = x_str.substr(i * 16, 16);
            x.array()[3 - i] = std::stoull(chunk, nullptr, 16);
        }

        Field mont_x;
        Field::mul(mont_x, x, Field::r_sqr());

        auto gec_rng = gec::make_gec_rng(std::mt19937_64{std::random_device{}()});
        if (!Point::lift_x(input_point, mont_x, is_odd, gec_rng))
        {
            throw std::runtime_error("Failed to lift x coordinate");
        }
    }
};

int main(int argc, char *argv[])
{
    try
    {
        if (argc != 4)
        {
            std::cerr << "Usage: " << argv[0]
                      << " <compressed_pubkey> <start:end> <batch_size>\n";
            return 1;
        }

        std::string range_str(argv[2]);
        size_t colon_pos = range_str.find(':');
        if (colon_pos == std::string::npos)
        {
            std::cerr << "Invalid range format. Use start:end\n";
            return 1;
        }

        uint64_t start = std::stoull(range_str.substr(0, colon_pos));
        uint64_t end = std::stoull(range_str.substr(colon_pos + 1));
        size_t batch_size = std::stoull(argv[3]);

        GPUPubkeySubtractor subtractor(argv[1], batch_size);
        subtractor.subtract_range(start, end);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}