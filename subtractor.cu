#include "subtractor.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <bigint/data/literal.hpp>
#include <bigint/data/array.hpp>
#include <bigint/mixin/division.hpp>
#include <bigint/mixin/add_sub.hpp>
#include <bigint/preset.hpp>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <thread>

namespace cg = cooperative_groups;
const std::string RANGE_START = "4000000000000000000000000000000000";
const std::string RANGE_END = "7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF";

__global__ void subtract_kernel(
    gec::curve::secp256k1::Curve<gec::curve::JacobianCurve> *points,
    const gec::curve::secp256k1::Curve<gec::curve::JacobianCurve> &input_point,
    const uint64_t *random_values,
    size_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    using JPoint = gec::curve::secp256k1::Curve<gec::curve::JacobianCurve>;
    using Scalar = gec::curve::secp256k1::Scalar;
    using Field = gec::curve::secp256k1::Field;

    // Initialize scalar directly without std::fill
    Scalar s;
    for (int i = 0; i < 4; ++i)
    {
        s.array()[i] = (i == 0) ? random_values[idx] : 0;
    }

    // Calculate generator point multiplication
    JPoint gen_mult;
    JPoint::mul(gen_mult, s, gec::curve::secp256k1::d_Gen);

    // Negate y coordinate for subtraction
    Field::neg(gen_mult.y(), gen_mult.y());

    // Add points (subtraction due to negation)
    JPoint::add(points[idx], input_point, gen_mult);
    JPoint::to_affine(points[idx]);
}

gec::curve::secp256k1::Scalar parse_hex_string(const std::string &hex)
{
    gec::curve::secp256k1::Scalar result;
    std::string padded_hex = std::string(64 - hex.length(), '0') + hex;

    for (int i = 0; i < 4; i++)
    {
        std::string chunk = padded_hex.substr(i * 16, 16);
        result.array()[3 - i] = std::stoull(chunk, nullptr, 16);
    }
    return result;
}

PubkeySubtractor::PubkeySubtractor(const std::string &pubkey, size_t optimal_batch_size)
    : start_time(std::chrono::steady_clock::now()),
      batch_size(optimal_batch_size)
{
    if (!parse_pubkey(pubkey))
    {
        throw std::runtime_error("Invalid public key format");
    }
    initialize_device();
}

PubkeySubtractor::~PubkeySubtractor()
{
    cleanup_device();
    if (bloom_initialized)
    {
        bloom_free(&h_bloom_filter1);
        bloom_free(&h_bloom_filter2);
        bloom_free(&h_bloom_filter3);
        bloom_initialized = false;
    }
}

void PubkeySubtractor::initialize_device()
{
    cudaMalloc(&d_points, batch_size * sizeof(Point));
    cudaMalloc(&d_random_values, batch_size * sizeof(uint64_t));

    cudaMallocHost(&h_points, batch_size * sizeof(Point));
    cudaMallocHost(&h_random_values, batch_size * sizeof(uint64_t));

    cudaStreamCreate(&compute_stream);
    cudaStreamCreate(&transfer_stream);
    cudaEventCreate(&compute_done);
    cudaEventCreate(&transfer_done);
}

void PubkeySubtractor::cleanup_device()
{
    if (d_points)
        cudaFree(d_points);
    if (d_random_values)
        cudaFree(d_random_values);
    if (h_points)
        cudaFreeHost(h_points);
    if (h_random_values)
        cudaFreeHost(h_random_values);

    if (compute_stream)
        cudaStreamDestroy(compute_stream);
    if (transfer_stream)
        cudaStreamDestroy(transfer_stream);
    if (compute_done)
        cudaEventDestroy(compute_done);
    if (transfer_done)
        cudaEventDestroy(transfer_done);

    d_points = nullptr;
    d_random_values = nullptr;
    h_points = nullptr;
    h_random_values = nullptr;
    compute_stream = nullptr;
    transfer_stream = nullptr;
    compute_done = nullptr;
    transfer_done = nullptr;
}

bool PubkeySubtractor::init_bloom_filters(const std::string &filename)
{
    std::cout << "Initializing bloom filters...\n";
    std::cout << "Setting up filters with parameters:\n"
              << "Filter 1: " << MAX_ENTRIES1 << " entries, FP rate: " << BLOOM1_FP_RATE << "\n"
              << "Filter 2: " << MAX_ENTRIES2 << " entries, FP rate: " << BLOOM2_FP_RATE << "\n"
              << "Filter 3: " << MAX_ENTRIES3 << " entries, FP rate: " << BLOOM3_FP_RATE << "\n";

    if (bloom_init2(&h_bloom_filter1, MAX_ENTRIES1, BLOOM1_FP_RATE) != 0 ||
        bloom_init2(&h_bloom_filter2, MAX_ENTRIES2, BLOOM2_FP_RATE) != 0 ||
        bloom_init2(&h_bloom_filter3, MAX_ENTRIES3, BLOOM3_FP_RATE) != 0)
    {
        std::cerr << "Failed to initialize bloom filter structures\n";
        return false;
    }
    bloom_initialized = true;
    std::cout << "Bloom filter structures initialized successfully\n";

    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open file: " << filename << "\n";
        return false;
    }
    std::cout << "Opened file: " << filename << "\n";

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::cout << "File size: " << file_size << " bytes\n";

    bool is_binary = file_size % 33 == 0;
    size_t entry_size = is_binary ? 33 : 66;
    size_t total_entries = file_size / entry_size;
    std::cout << "Format: " << (is_binary ? "Binary" : "Hex") << " format detected\n"
              << "Entry size: " << entry_size << " bytes\n"
              << "Total entries to process: " << total_entries << "\n";

    unsigned int num_threads = std::thread::hardware_concurrency();
    std::cout << "Initializing " << num_threads << " worker threads\n";

    std::vector<std::thread> threads;
    std::mutex file_mutex;
    std::atomic<size_t> entries_processed{0};
    std::vector<WorkerBuffer> buffers(num_threads);

    std::cout << "Starting bloom filter population...\n";
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

            entries_processed += entries;
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
            if (elapsed > 0)
            {
                double rate = entries_processed / static_cast<double>(elapsed);
                double percent = (static_cast<double>(entries_processed) / total_entries) * 100;
                printf("\rProgress: %.1f%% - Processed %zu/%zu entries (%.2f entries/sec)...",
                       percent, entries_processed.load(), total_entries, rate);
                fflush(stdout);
            }
        }
    };

    for (unsigned int i = 0; i < num_threads; i++)
    {
        threads.emplace_back(worker, i);
    }

    for (auto &thread : threads)
    {
        thread.join();
    }

    printf("\nCompleted loading %zu entries into bloom filters\n", entries_processed.load());

    auto end_time = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    double final_rate = entries_processed / static_cast<double>(total_time);

    std::cout << "Bloom filter population complete\n"
              << "Time taken: " << total_time << " seconds\n"
              << "Final processing rate: " << final_rate << " entries/sec\n"
              << "Memory used per filter:\n"
              << "Filter 1: " << (h_bloom_filter1.bytes / 1024 / 1024) << " MB\n"
              << "Filter 2: " << (h_bloom_filter2.bytes / 1024 / 1024) << " MB\n"
              << "Filter 3: " << (h_bloom_filter3.bytes / 1024 / 1024) << " MB\n";

    return true;
}

void PubkeySubtractor::process_entries(const char *data, size_t num_entries, bool is_binary)
{
    std::vector<unsigned char> x_coord(32);
    const size_t binary_size = 33;
    const size_t hex_size = 66;

    for (size_t i = 0; i < num_entries; i++)
    {
        if (is_binary)
        {
            const unsigned char *pubkey_data = reinterpret_cast<const unsigned char *>(data + i * binary_size);
            std::memcpy(x_coord.data(), pubkey_data + 1, 32);
        }
        else
        {
            const char *hex_str = data + i * hex_size;
            for (size_t j = 0; j < 32; j++)
            {
                unsigned int byte;
                std::sscanf(hex_str + 2 + j * 2, "%02x", &byte);
                x_coord[j] = static_cast<unsigned char>(byte);
            }
        }

        bloom_add(&h_bloom_filter1, reinterpret_cast<const char *>(x_coord.data()), PUBKEY_PREFIX_LENGTH);

        XXH64_hash_t hash1 = XXH64(x_coord.data(), 32, 0x1234);
        bloom_add(&h_bloom_filter2, reinterpret_cast<const char *>(&hash1), sizeof(hash1));

        XXH64_hash_t hash2 = XXH64(x_coord.data(), 32, 0x5678);
        bloom_add(&h_bloom_filter3, reinterpret_cast<const char *>(&hash2), sizeof(hash2));
    }
}

bool PubkeySubtractor::check_bloom_filters(const std::string &compressed)
{
    std::vector<unsigned char> x_coord(32);
    for (size_t i = 0; i < 32; i++)
    {
        unsigned int byte;
        std::sscanf(compressed.substr(2 + i * 2, 2).c_str(), "%02x", &byte);
        x_coord[i] = static_cast<unsigned char>(byte);
    }

    if (!bloom_check(&h_bloom_filter1, reinterpret_cast<const char *>(x_coord.data()), PUBKEY_PREFIX_LENGTH))
    {
        return false;
    }

    XXH64_hash_t hash1 = XXH64(x_coord.data(), 32, 0x1234);
    if (!bloom_check(&h_bloom_filter2, reinterpret_cast<const char *>(&hash1), sizeof(hash1)))
    {
        return false;
    }

    XXH64_hash_t hash2 = XXH64(x_coord.data(), 32, 0x5678);
    return bloom_check(&h_bloom_filter3, reinterpret_cast<const char *>(&hash2), sizeof(hash2));
}

void PubkeySubtractor::subtract_range(const Scalar &start, const Scalar &end, std::atomic<bool> &should_stop)
{
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    if (!d_points || !d_random_values || !h_points || !h_random_values)
    {
        std::cerr << "Device resources not properly initialized\n";
        return;
    }

    const size_t block_size = 256;
    const size_t num_blocks = (batch_size + block_size - 1) / block_size;
    std::mt19937_64 rng(std::random_device{}());

    // Create ranges for each limb
    std::uniform_int_distribution<uint64_t> dist0(start.array()[0], end.array()[0]);
    std::uniform_int_distribution<uint64_t> dist1(0, UINT64_MAX);
    std::uniform_int_distribution<uint64_t> dist2(0, UINT64_MAX);
    std::uniform_int_distribution<uint64_t> dist3(0, UINT64_MAX);

    std::cout << "Range: " << std::hex
              << "Start: " << std::setfill('0') << std::setw(16)
              << start.array()[3] << start.array()[2] << start.array()[1] << start.array()[0] << "\n"
              << "End: " << std::setfill('0') << std::setw(16)
              << end.array()[3] << end.array()[2] << end.array()[1] << end.array()[0] << std::dec << "\n"
              << "Batch size: " << batch_size << ", Blocks: " << num_blocks << "x" << block_size << "\n";

    while (!should_stop)
    {
        Scalar current;
        for (size_t i = 0; i < batch_size; ++i)
        {
            h_random_values[i] = dist0(rng);

            // If at start range, ensure higher limbs are within range
            if (h_random_values[i] == start.array()[0])
            {
                current.array()[1] = start.array()[1] + (rng() % (UINT64_MAX - start.array()[1]));
                current.array()[2] = start.array()[2];
                current.array()[3] = start.array()[3];
            }
            // If at end range, ensure higher limbs don't exceed end
            else if (h_random_values[i] == end.array()[0])
            {
                current.array()[1] = rng() % (end.array()[1] + 1);
                current.array()[2] = rng() % (end.array()[2] + 1);
                current.array()[3] = rng() % (end.array()[3] + 1);
            }
            // In middle of range, use full distribution
            else
            {
                current.array()[1] = dist1(rng);
                current.array()[2] = dist2(rng);
                current.array()[3] = dist3(rng);
            }
        }

        cudaMemcpyAsync(d_random_values, h_random_values,
                        batch_size * sizeof(uint64_t),
                        cudaMemcpyHostToDevice, compute_stream);

        subtract_kernel<<<num_blocks, block_size, 0, compute_stream>>>(
            d_points,
            h_input_point,
            d_random_values,
            batch_size);

        cudaEventRecord(compute_done, compute_stream);
        cudaStreamWaitEvent(transfer_stream, compute_done);

        cudaMemcpyAsync(h_points, d_points,
                        batch_size * sizeof(Point),
                        cudaMemcpyDeviceToHost, transfer_stream);

        cudaEventRecord(transfer_done, transfer_stream);
        cudaEventSynchronize(transfer_done);

        for (size_t i = 0; i < batch_size; i++)
        {
            const auto &point = h_points[i];
            std::stringstream ss;
            ss << std::hex << std::setfill('0');

            Field y_norm;
            Field::from_montgomery(y_norm, point.y());
            bool is_odd = (y_norm.array()[0] & 1) != 0;
            ss << (is_odd ? "03" : "02");

            Field x_norm;
            Field::from_montgomery(x_norm, point.x());
            for (size_t j = Field::LimbN; j > 0; --j)
            {
                ss << std::setw(16) << x_norm.array()[j - 1];
            }
            generated_pubkey = ss.str();

            ss.str("");
            ss.clear();
            ss << std::hex << std::setw(16) << h_random_values[i];
            current_subtraction_value = ss.str();

            if (check_bloom_filters(generated_pubkey))
            {
                save_match(generated_pubkey, h_random_values[i]);
            }
        }

        attempts += batch_size;
        if (attempts.load() % (batch_size * 10) == 0)
        {
            report_status();
        }
    }

    cudaDeviceSynchronize();
    report_status(true);
}

void PubkeySubtractor::save_match(const std::string &pubkey, uint64_t value)
{
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << "\nPotential match found!\n"
              << "Public key: " << pubkey << "\n"
              << "Subtraction value: " << std::hex << value << std::dec << "\n";

    std::ofstream match_file("matches.txt", std::ios::app);
    if (match_file)
    {
        match_file << "Public Key: " << pubkey << "\n"
                   << "Subtraction: " << std::hex << std::setfill('0') << std::setw(16) << value << std::dec << "\n\n";
    }
}

void PubkeySubtractor::report_status(bool final)
{
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
    uint64_t current_attempts = attempts.load();
    double kps = elapsed > 0 ? static_cast<double>(current_attempts) / elapsed : 0;

    std::lock_guard<std::mutex> lock(cout_mutex);
    if (final)
    {
        std::cout << "\nCompleted with " << std::fixed << std::setprecision(2)
                  << kps << " k/s" << std::endl;
    }
    else
    {
        std::cout << "\rPK: " << generated_pubkey
                  << " SUB: " << current_subtraction_value
                  << " Speed: " << std::fixed << std::setprecision(2)
                  << kps << " k/s" << std::flush;
    }
}

bool PubkeySubtractor::parse_pubkey(const std::string &pubkey)
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
    return Point::lift_x(h_input_point, mont_x, is_odd, gec_rng);
}
