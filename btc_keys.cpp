#include <gec/curve/secp256k1.hpp>
#include <gec/bigint/mixin/random.hpp>
#include <gec/utils/basic.hpp>
#include <gec/utils/crtp.hpp>
#include "bloom.h"
#include "xxhash.h"
#include <random>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <fstream>
#include <cstring>
#include <chrono>

// Constants
constexpr size_t MAX_ENTRIES1 = 100000000;
constexpr size_t MAX_ENTRIES2 = 80000000;
constexpr size_t MAX_ENTRIES3 = 60000000;
constexpr double BLOOM1_FP_RATE = 0.0001;
constexpr double BLOOM2_FP_RATE = 0.00001;
constexpr double BLOOM3_FP_RATE = 0.000001;
constexpr size_t PUBKEY_PREFIX_LENGTH = 6;
constexpr size_t BATCH_SIZE = 1024;
constexpr size_t BUFFER_SIZE = 1024 * 1024; // 1MB buffer per thread

// CustomRng from CORRECT_Pri_2_PubMulti.cpp
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

class BloomFilters
{
public:
    bloom bloom_filter1;
    bloom bloom_filter2;
    bloom bloom_filter3;
    bool initialized{false};
    std::atomic<size_t> entries_processed{0};
    std::mutex file_mutex;

    ~BloomFilters()
    {
        cleanup();
    }

    void cleanup()
    {
        if (initialized)
        {
            bloom_free(&bloom_filter1);
            bloom_free(&bloom_filter2);
            bloom_free(&bloom_filter3);
            initialized = false;
        }
    }

    bool init()
    {
        cleanup();
        if (bloom_init2(&bloom_filter1, MAX_ENTRIES1, BLOOM1_FP_RATE) != 0 ||
            bloom_init2(&bloom_filter2, MAX_ENTRIES2, BLOOM2_FP_RATE) != 0 ||
            bloom_init2(&bloom_filter3, MAX_ENTRIES3, BLOOM3_FP_RATE) != 0)
        {
            return false;
        }
        initialized = true;
        return true;
    }

    struct WorkerBuffer
    {
        std::vector<char> data;
        size_t used{0};

        WorkerBuffer() : data(BUFFER_SIZE) {}

        void reset()
        {
            used = 0;
        }
    };

    void process_entries(const char *data, size_t num_entries, bool is_binary)
    {
        std::vector<unsigned char> x_coord(32);
        size_t entry_size = is_binary ? 33 : 66;

        for (size_t i = 0; i < num_entries; i++)
        {
            if (is_binary)
            {
                const unsigned char *pubkey_data = reinterpret_cast<const unsigned char *>(data + i * 33);
                std::memcpy(x_coord.data(), pubkey_data + 1, 32);
            }
            else
            {
                const char *hex_str = data + i * 66;
                for (size_t j = 0; j < 32; j++)
                {
                    unsigned int byte;
                    std::sscanf(hex_str + 2 + j * 2, "%02x", &byte);
                    x_coord[j] = static_cast<unsigned char>(byte);
                }
            }

            bloom_add(&bloom_filter1, reinterpret_cast<const char *>(x_coord.data()), PUBKEY_PREFIX_LENGTH);

            XXH64_hash_t hash1 = XXH64(x_coord.data(), 32, 0x1234);
            bloom_add(&bloom_filter2, reinterpret_cast<const char *>(&hash1), sizeof(hash1));

            XXH64_hash_t hash2 = XXH64(x_coord.data(), 32, 0x5678);
            bloom_add(&bloom_filter3, reinterpret_cast<const char *>(&hash2), sizeof(hash2));
        }
        entries_processed += num_entries;
    }

    bool load_pubkeys(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file)
            return false;

        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        bool is_binary = file_size % 33 == 0;
        size_t entry_size = is_binary ? 33 : 66;
        size_t total_entries = file_size / entry_size;

        unsigned int num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        std::vector<WorkerBuffer> buffers(num_threads);

        auto start_time = std::chrono::steady_clock::now();

        // Worker thread implementation
        auto worker = [&](int thread_id)
        {
            WorkerBuffer &buffer = buffers[thread_id];
            while (true)
            {
                size_t bytes_read;
                {
                    std::lock_guard<std::mutex> lock(file_mutex);
                    if (!file || file.eof())
                        break;

                    file.read(buffer.data.data(), buffer.data.size());
                    bytes_read = file.gcount();
                }

                if (bytes_read == 0)
                    break;

                size_t entries = bytes_read / entry_size;
                process_entries(buffer.data.data(), entries, is_binary);
                buffer.reset();

                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
                if (elapsed > 0)
                {
                    double rate = entries_processed / static_cast<double>(elapsed);
                    printf("\rProcessed %zu/%zu entries (%.2f entries/sec)...",
                           entries_processed.load(), total_entries, rate);
                    fflush(stdout);
                }
            }
        };

        // Launch worker threads
        for (unsigned int i = 0; i < num_threads; i++)
        {
            threads.emplace_back(worker, i);
        }

        // Join threads
        for (auto &thread : threads)
        {
            thread.join();
        }

        printf("\nCompleted loading %zu entries\n", entries_processed.load());
        return true;
    }

    bool check(const std::string &pubkey)
    {
        if (!initialized || pubkey.length() < 66)
            return false;

        std::vector<unsigned char> x_coord(32);
        for (size_t i = 0; i < 32; i++)
        {
            unsigned int byte;
            std::sscanf(pubkey.substr(2 + i * 2, 2).c_str(), "%02x", &byte);
            x_coord[i] = static_cast<unsigned char>(byte);
        }

        if (!bloom_check(&bloom_filter1, reinterpret_cast<const char *>(x_coord.data()), PUBKEY_PREFIX_LENGTH))
        {
            return false;
        }

        XXH64_hash_t hash1 = XXH64(x_coord.data(), 32, 0x1234);
        if (!bloom_check(&bloom_filter2, reinterpret_cast<const char *>(&hash1), sizeof(hash1)))
        {
            return false;
        }

        XXH64_hash_t hash2 = XXH64(x_coord.data(), 32, 0x5678);
        return bloom_check(&bloom_filter3, reinterpret_cast<const char *>(&hash2), sizeof(hash2));
    }
};

class PubkeySubtractor
{
    using Point = gec::curve::secp256k1::Curve<>;
    using Field = gec::curve::secp256k1::Field;
    using Scalar = gec::curve::secp256k1::Scalar;

    Point input_point;
    std::atomic<bool> should_stop{false};
    std::mutex cout_mutex;
    BloomFilters blooms;
    std::atomic<uint64_t> attempts{0};
    std::chrono::steady_clock::time_point start_time;

public:
    PubkeySubtractor(const std::string &pubkey)
    {
        if (!parse_pubkey(pubkey))
        {
            throw std::runtime_error("Invalid public key format");
        }
        start_time = std::chrono::steady_clock::now();
    }

    bool init_blooms(const std::string &filename)
    {
        if (!blooms.init())
        {
            return false;
        }
        return blooms.load_pubkeys(filename);
    }

    bool parse_pubkey(const std::string &pubkey)
    {
        if (pubkey.length() != 66 || pubkey[0] != '0' || (pubkey[1] != '2' && pubkey[1] != '3'))
        {
            return false;
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

        auto gec_rng = gec::GecRng<CustomRng>(CustomRng());
        return Point::lift_x(input_point, mont_x, is_odd, gec_rng);
    }

    void subtract_range(const Scalar &start, const Scalar &end)
    {
        unsigned int num_threads = 2;
        std::vector<std::thread> threads;

        // Start worker threads
        for (unsigned int i = 0; i < num_threads; i++)
        {
            threads.emplace_back([this, &start, &end]()
                                 { worker_thread(start, end); });
        }

        // Start status thread
        threads.emplace_back([this]()
                             {
            while (!should_stop) {
                report_status();
                std::this_thread::sleep_for(std::chrono::seconds(1));
            } });

        std::cout << "\nPress Enter to stop..." << std::endl;
        std::cin.get();
        should_stop = true;

        for (auto &thread : threads)
        {
            thread.join();
        }

        report_status(true);
    }

private:
    void report_status(bool final = false)
    {
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
        uint64_t current_attempts = attempts.load();

        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cout << "\rAttempts: " << current_attempts;
        if (elapsed > 0)
        {
            double rate = static_cast<double>(current_attempts) / elapsed;
            std::cout << " | Rate: " << std::fixed << std::setprecision(2)
                      << rate << " attempts/sec";
        }
        std::cout << std::string(20, ' ') << (final ? "\n" : "\r") << std::flush;
    }

    void worker_thread(const Scalar &start, const Scalar &end)
    {
        CustomRng rng_base;
        auto gec_rng = gec::GecRng<CustomRng>(rng_base);

        Point gen_mult, result;
        Scalar s;

        while (!should_stop)
        {
            // Generate random scalar between start and end
            s.array()[3] = 0;
            s.array()[2] = start.array()[2] + (rng_base.operator()<uint64_t>() %
                                               (end.array()[2] - start.array()[2]));
            s.array()[1] = rng_base.operator()<uint64_t>();
            s.array()[0] = rng_base.operator()<uint64_t>();

            // Use GEC's optimized point arithmetic
            Point::mul(gen_mult, s, gec::curve::secp256k1::Gen);
            Field::neg(gen_mult.y(), gen_mult.y());
            Point::add(result, input_point, gen_mult);
            Point::to_affine(result);

            // Get compressed public key
            std::stringstream ss;
            ss << std::hex << std::setfill('0');

            Field y_norm;
            Field::from_montgomery(y_norm, result.y());
            bool is_odd = (y_norm.array()[0] & 1) != 0;
            ss << (is_odd ? "03" : "02");

            Field x_norm;
            Field::from_montgomery(x_norm, result.x());
            const auto &arr = x_norm.array();
            for (size_t i = Field::LimbN; i > 0; --i)
            {
                ss << std::setw(16) << arr[i - 1];
            }
            std::string compressed = ss.str();

            attempts++;

            {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cout << "\r" << compressed << " - ";
                for (int i = 3; i >= 0; --i)
                {
                    std::cout << std::hex << std::setfill('0') << std::setw(16)
                              << s.array()[i];
                }
                std::cout << std::dec << std::flush;
            }

            // Check result against bloom filters
            if (blooms.check(compressed))
            {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cout << "\nPotential match found!\n"
                          << "Public key: " << compressed << "\n"
                          << "Subtraction value: ";
                for (int i = 3; i >= 0; --i)
                {
                    std::cout << std::hex << std::setfill('0') << std::setw(16)
                              << s.array()[i];
                }
                std::cout << std::dec << std::endl;

                // Save match to file
                std::ofstream match_file("matches.txt", std::ios::app);
                if (match_file)
                {
                    match_file << "Public Key: " << compressed << "\n"
                               << "Subtraction: ";
                    for (int i = 3; i >= 0; --i)
                    {
                        match_file << std::hex << std::setfill('0') << std::setw(16)
                                   << s.array()[i];
                    }
                    match_file << std::dec << "\n\n";
                }
            }
        }
    }
};

int main(int argc, char *argv[])
{
    try
    {
        if (argc != 5 || std::string(argv[3]) != "-f")
        {
            std::cout << "Usage: " << argv[0] << " <compressed_pubkey> <start:end> -f <pubkeys.bin>\n"
                      << "Example: " << argv[0] << " 02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16"
                      << " 21778071482940061661655974875633165533184:43556142965880123323311949751266331066367"
                      << " -f scanned_pubkeys.bin" << std::endl;
            return 1;
        }

        const std::string pubkey = argv[1];
        const std::string range_str = argv[2];
        const std::string bloom_file = argv[4];

        size_t colon_pos = range_str.find(':');
        if (colon_pos == std::string::npos)
        {
            std::cerr << "Invalid range format. Use start:end" << std::endl;
            return 1;
        }

        using Scalar = gec::curve::secp256k1::Scalar;
        Scalar start, end;

        try
        {
            // Initialize scalars
            for (int i = 0; i < 4; i++)
            {
                start.array()[i] = 0;
                end.array()[i] = 0;
            }

            // Parse large decimal strings to limbs
            std::string start_str = range_str.substr(0, colon_pos);
            std::string end_str = range_str.substr(colon_pos + 1);

            // Convert to limbs, handling chunks of 16 characters (64 bits)
            size_t pos = 0;
            int limb = 0;
            while (pos < start_str.length() && limb < 4)
            {
                size_t chunk_size = std::min<size_t>(16, start_str.length() - pos);
                std::string chunk = start_str.substr(pos, chunk_size);
                start.array()[limb++] = std::stoull(chunk);
                pos += chunk_size;
            }

            pos = 0;
            limb = 0;
            while (pos < end_str.length() && limb < 4)
            {
                size_t chunk_size = std::min<size_t>(16, end_str.length() - pos);
                std::string chunk = end_str.substr(pos, chunk_size);
                end.array()[limb++] = std::stoull(chunk);
                pos += chunk_size;
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error parsing range values: " << e.what() << std::endl;
            std::cerr << "Please provide valid decimal numbers for the range" << std::endl;
            return 1;
        }

        PubkeySubtractor subtractor(pubkey);
        std::cout << "Initializing bloom filters from: " << bloom_file << std::endl;

        if (!subtractor.init_blooms(bloom_file))
        {
            std::cerr << "Failed to initialize bloom filters" << std::endl;
            return 1;
        }

        std::cout << "\nConfiguration:\n"
                  << "Public Key: " << pubkey << "\n"
                  << "Range Start: " << range_str.substr(0, colon_pos) << "\n"
                  << "Range End: " << range_str.substr(colon_pos + 1) << "\n"
                  << "Bloom Filter File: " << bloom_file << "\n"
                  << "Using Threads: " << std::thread::hardware_concurrency() << "\n\n";

        signal(SIGINT, [](int)
               {
            std::cout << "\nReceived interrupt signal. Shutting down...\n";
            std::exit(0); });

        subtractor.subtract_range(start, end);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nProgram completed successfully." << std::endl;
    return 0;
}