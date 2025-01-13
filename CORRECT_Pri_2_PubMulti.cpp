#include <gec/curve/secp256k1.hpp>
#include <gec/bigint/mixin/random.hpp>
#include <random>
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>

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

const std::string TARGET_PUBKEY = "04145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16667a05e9a1bdd6f70142b66558bd12ce2c0f9cbc7001b20c8a6a109c80dc5330";
const std::string RANGE_START = "4000000000000000000000000000000000";
const std::string RANGE_END = "7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF";

std::mutex cout_mutex;

gec::curve::secp256k1::Scalar parse_hex_string(const std::string &hex)
{
    gec::curve::secp256k1::Scalar result;
    for (int i = 0; i < 4; i++)
    {
        std::string chunk = hex.substr(i * 16, 16);
        result.array()[3 - i] = std::stoull(chunk, nullptr, 16);
    }
    return result;
}

void generate_keypair_thread(int thread_id, std::atomic<bool> &running)
{
    using namespace gec::curve::secp256k1;
    using Point = Curve<>;

    CustomRng rng_base;
    auto rng = gec::GecRng<CustomRng>(rng_base);

    while (running)
    {
        Scalar private_key;

        private_key.array()[3] = 0;
        private_key.array()[2] = 0x40ULL + (rng_base.operator()<uint64_t>() % 0x40ULL); // Only first 2 hex digits between 40-7F
        private_key.array()[1] = rng_base.operator()<uint64_t>();
        private_key.array()[0] = rng_base.operator()<uint64_t>();

        std::string private_key_str = to_hex(private_key);

        // Generate the public key
        Point public_key;
        Point::mul(public_key, private_key, Gen);
        Point::to_affine(public_key);

        Field x_coord = public_key.x();
        Field y_coord = public_key.y();
        Field::from_montgomery(x_coord, x_coord);
        Field::from_montgomery(y_coord, y_coord);

        std::string pub_key = "04" + to_hex(x_coord) + to_hex(y_coord);
        std::string pub_key_short = pub_key.substr(0, 66);

        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "\rPri: " << private_key_str << " Pubkey: " << pub_key_short << std::flush;

            if (pub_key == TARGET_PUBKEY)
            {
                std::cout << "\nMatch found!\nPrivate Key: " << private_key_str << "\nPublic Key: " << pub_key << std::endl;
                running = false;
                break;
            }
        }
    }
}

int main()
{
    try
    {
        unsigned int num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        std::atomic<bool> running(true);

        for (unsigned int i = 0; i < num_threads; i++)
        {
            threads.emplace_back(generate_keypair_thread, i, std::ref(running));
        }

        std::cin.get();
        running = false;

        for (auto &thread : threads)
        {
            thread.join();
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}