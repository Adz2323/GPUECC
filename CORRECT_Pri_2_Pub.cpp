#include <gec/curve/secp256k1.hpp>
#include <gec/bigint/mixin/random.hpp>
#include <random>
#include <iostream>
#include <iomanip>
#include <sstream>

using namespace gec::bigint::literal;

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

template <typename T>
std::string to_hex(const T &value)
{
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    const auto &arr = value.array();
    for (size_t i = T::LimbN; i > 0; --i)
    {
        ss << std::setw(16) << arr[i - 1];
    }
    return ss.str();
}

void generate_keypair()
{
    using namespace gec::curve::secp256k1;
    using Point = Curve<>;

    CustomRng rng_base;
    auto rng = gec::GecRng<CustomRng>(rng_base);

    // Generate private key
    Scalar private_key;
    do
    {
        Scalar::sample(private_key, rng);
    } while (private_key >= Scalar::mod() || private_key.is_zero());

    // Generate public key
    Point public_key;
    Point::mul(public_key, private_key, Gen); // Don't convert private key to Montgomery form
    Point::to_affine(public_key);

    Field x_coord = public_key.x();
    Field y_coord = public_key.y();
    Field::from_montgomery(x_coord, x_coord);
    Field::from_montgomery(y_coord, y_coord);

    std::cout << "Private key: " << to_hex(private_key) << std::endl;
    std::cout << "Public key (uncompressed):\n04"
              << to_hex(x_coord)
              << to_hex(y_coord) << std::endl;
}

int main()
{
    try
    {
        generate_keypair();
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}