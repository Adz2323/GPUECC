#include "subtractor.cuh"
#include <bigint/data/literal.hpp>
#include <bigint/preset.hpp>
#include <utils/basic.hpp>
#include <utils.hpp>
#include <curve.hpp>
#include <bigint.hpp>
#include <iostream>
#include <iomanip>

using namespace gec::curve::secp256k1;
using namespace gec::bigint::literal;

namespace gec
{
    template <>
    struct GecRng<CustomRng>
    {
        CustomRng rng;
        GecRng(CustomRng r) : rng(r) {}

        template <typename T>
        GEC_HD T sample() { return rng.template operator()<T>(); }

        template <typename T>
        GEC_HD T sample(T higher) { return rng(higher); }

        template <typename T>
        GEC_HD T sample(T lower, T higher) { return rng(lower, higher); }
    };
}

namespace
{
    void check_cuda_error(cudaError_t err, const char *msg)
    {
        if (err != cudaSuccess)
        {
            std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
            exit(1);
        }
    }

    template <typename Core>
    void print_bigint(const Core &num, const char *prefix = "")
    {
        std::cout << prefix << std::hex;
        for (int i = Core::LimbN - 1; i >= 0; i--)
        {
            std::cout << std::setfill('0') << std::setw(16) << num.array()[i];
        }
        std::cout << std::dec << std::endl;
    }

    bool parse_compressed_pubkey(const std::string &pubkey_hex, Curve<> &out_point)
    {
        if (pubkey_hex.length() != 66 || pubkey_hex[0] != '0' ||
            (pubkey_hex[1] != '2' && pubkey_hex[1] != '3'))
        {
            return false;
        }

        bool is_odd = (pubkey_hex[1] == '3');
        std::string x_hex = pubkey_hex.substr(2);

        Field x;
        for (int i = 0; i < 4; i++)
        {
            std::string chunk = x_hex.substr(i * 16, 16);
            x.array()[3 - i] = std::stoull(chunk, nullptr, 16);
        }

        Field mont_x;
        Field::to_montgomery(mont_x, x);

        CustomRng rng;
        auto gec_rng = gec::GecRng<CustomRng>(rng);
        return Curve<>::lift_x(out_point, mont_x, is_odd, gec_rng);
    }

    void print_point_compressed(const Curve<> &point)
    {
        Field x = point.x();
        Field y = point.y();
        Field z = point.z();

        Field::from_montgomery(x, x);
        Field::from_montgomery(y, y);
        Field::from_montgomery(z, z);

        bool is_odd = (y.array()[0] & 1) != 0;
        std::cout << (is_odd ? "03" : "02");
        print_bigint(x);
    }

    Scalar parse_scalar(const std::string &dec_str)
    {
        using BigInt = gec::bigint::BigintBE<uint64_t, 4>;
        BigInt value;
        value.set_zero();
        BigInt base, temp;
        base.set_one();

        for (int i = dec_str.length() - 1; i >= 0; i--)
        {
            temp = value;
            uint8_t digit = dec_str[i] - '0';
            for (uint8_t j = 0; j < digit; j++)
            {
                BigInt::add(value, temp, base);
            }
            if (i > 0)
            {
                temp = base;
                for (int j = 0; j < 10; j++)
                {
                    BigInt::add(base, base, temp);
                }
            }
        }

        Scalar result;
        for (int i = 0; i < 4; i++)
        {
            result.array()[i] = value.array()[i];
        }
        return result;
    }
}

int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <pubkey_to_subtract_from>\n";
        return 1;
    }

    const std::string pubkey = argv[1];
    const std::string subtract_decimal = "30585208500956698822931703462279545183414";

    Curve<> input_point;
    if (!parse_compressed_pubkey(pubkey, input_point))
    {
        std::cerr << "Failed to parse pubkey\n";
        return 1;
    }

    Scalar subtract_value = parse_scalar(subtract_decimal);

    std::cout << "Input pubkey: " << pubkey << "\n";
    std::cout << "Subtracting value: ";
    print_bigint(subtract_value);

    // CPU computation
    Curve<> gen_mult, cpu_result;
    gen_mult = Gen;

    // Convert scalar to Montgomery form
    Scalar mont_subtract = subtract_value;
    Field::to_montgomery(mont_subtract, mont_subtract);

    Curve<>::mul(gen_mult, mont_subtract, gen_mult);
    Field::neg(gen_mult.y(), gen_mult.y());
    Curve<>::add(cpu_result, input_point, gen_mult);
    Curve<>::to_affine(cpu_result);

    std::cout << "\nCPU result: ";
    print_point_compressed(cpu_result);

    // GPU computation
    const size_t batch_size = 1;
    SubtractorBatch batch(batch_size);

    check_cuda_error(
        initialize_subtractor(batch, input_point, mont_subtract, mont_subtract),
        "Init failed");

    dim3 block(1);
    dim3 grid(1);
    check_cuda_error(launch_subtraction(batch, grid, block), "Launch failed");
    check_cuda_error(cudaDeviceSynchronize(), "Sync failed");

    SubtractionResult gpu_result;
    check_cuda_error(
        cudaMemcpy(&gpu_result, batch.results(), sizeof(SubtractionResult), cudaMemcpyDeviceToHost),
        "Copy failed");

    std::cout << "\nGPU result: ";
    print_point_compressed(gpu_result.result);

    return 0;
}
