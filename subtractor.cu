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
    const gec::curve::secp256k1::Curve<gec::curve::JacobianCurve> *input_point,
    const gec::curve::secp256k1::Scalar start,
    const gec::curve::secp256k1::Scalar step,
    const size_t batch_size,
    gec::curve::secp256k1::Curve<gec::curve::JacobianCurve> *results,
    gec::curve::secp256k1::Scalar *scalars)
{
    using namespace gec::curve::secp256k1;
    using JPoint = Curve<gec::curve::JacobianCurve>;

    cg::grid_group grid = cg::this_grid();
    const size_t idx = grid.thread_rank();

    if (idx >= batch_size)
        return;

    // Calculate scalar for this thread
    Scalar current = start;
    for (size_t i = 0; i < idx; i++)
    {
        current.add(current, step);
    }

    // Calculate generator point multiplication
    JPoint gen_mult;
    JPoint::mul(gen_mult, current, d_Gen);

    // Negate y coordinate for subtraction
    Field::neg(gen_mult.y(), gen_mult.y());

    // Add to input point
    JPoint::add(results[idx], *input_point, gen_mult);
    JPoint::to_affine(results[idx]);

    // Store scalar
    scalars[idx] = current;

    grid.sync();
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
    cudaStreamCreate(&compute_stream);
    cudaStreamCreate(&transfer_stream);
    cudaEventCreate(&compute_done);
    cudaEventCreate(&transfer_done);

    cudaMalloc(&d_input_point, sizeof(CurvePoint));
    cudaMalloc(&d_results, batch_size * sizeof(CurvePoint));
    cudaMalloc(&d_scalars, batch_size * sizeof(Scalar));

    cudaMemcpy(d_input_point, &h_input_point, sizeof(CurvePoint), cudaMemcpyHostToDevice);
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

void PubkeySubtractor::cleanup_device()
{
    if (d_input_point)
        cudaFree(d_input_point);
    if (d_results)
        cudaFree(d_results);
    if (d_scalars)
        cudaFree(d_scalars);

    if (compute_stream)
        cudaStreamDestroy(compute_stream);
    if (transfer_stream)
        cudaStreamDestroy(transfer_stream);
    if (compute_done)
        cudaEventDestroy(compute_done);
    if (transfer_done)
        cudaEventDestroy(transfer_done);

    d_input_point = nullptr;
    d_results = nullptr;
    d_scalars = nullptr;
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

    if (!d_input_point || !d_results || !d_scalars || !compute_stream || !transfer_stream)
    {
        std::cerr << "Device resources not properly initialized\n";
        return;
    }

    // Calculate block configuration
    const size_t block_size = 256;
    int max_blocks_per_sm;
    cudaDeviceGetAttribute(&max_blocks_per_sm, cudaDevAttrMaxBlocksPerMultiprocessor, 0);

    // Limit blocks to 16 per SM for better occupancy
    const size_t total_sms = deviceProp.multiProcessorCount;
    const size_t max_blocks = std::min(16U * total_sms, static_cast<size_t>(max_blocks_per_sm * total_sms));
    const size_t num_blocks = std::min((batch_size + block_size - 1) / block_size, max_blocks);
    const size_t actual_batch_size = num_blocks * block_size;

    std::cout << "Grid configuration: " << num_blocks << " blocks, "
              << block_size << " threads per block ("
              << actual_batch_size << " total threads)\n";

    // Allocate host buffers
    CurvePoint *h_results;
    Scalar *h_scalars;
    cudaHostAlloc(&h_results, actual_batch_size * sizeof(CurvePoint), cudaHostAllocDefault);
    cudaHostAlloc(&h_scalars, actual_batch_size * sizeof(Scalar), cudaHostAllocDefault);

    CustomRng rng_base;
    Scalar current, step;
    for (int i = 0; i < 4; i++)
        step.array()[i] = 0;
    step.array()[0] = 1;

    std::cout << "Range: " << std::hex
              << "Start: " << start.array()[3] << start.array()[2] << start.array()[1] << start.array()[0] << "\n"
              << "End: " << end.array()[3] << end.array()[2] << end.array()[1] << end.array()[0] << std::dec << "\n"
              << "Batch size: " << actual_batch_size << ", Blocks: " << num_blocks << "x" << block_size << "\n";

    while (!should_stop)
    {
        // Generate random scalar
        current.array()[3] = start.array()[3] + (rng_base.operator()<uint64_t>() % (end.array()[3] - start.array()[3] + 1));
        if (current.array()[3] == end.array()[3])
        {
            current.array()[2] = start.array()[2] + (rng_base.operator()<uint64_t>() % (end.array()[2] - start.array()[2] + 1));
        }
        else if (current.array()[3] == start.array()[3])
        {
            current.array()[2] = start.array()[2] + rng_base.operator()<uint64_t>();
        }
        else
        {
            current.array()[2] = rng_base.operator()<uint64_t>();
        }
        current.array()[1] = rng_base.operator()<uint64_t>();
        current.array()[0] = rng_base.operator()<uint64_t>();

        void *args[] = {(void *)&d_input_point, (void *)&current, (void *)&step,
                        (void *)&actual_batch_size, (void *)&d_results, (void *)&d_scalars};

        cudaStreamSynchronize(compute_stream);
        cudaError_t kernelError = cudaLaunchCooperativeKernel(
            (void *)subtract_kernel, num_blocks, block_size, args, 0, compute_stream);

        if (kernelError != cudaSuccess)
        {
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(kernelError) << "\n";
            cudaGetLastError();
            continue;
        }

        cudaEventRecord(compute_done, compute_stream);
        if (cudaEventSynchronize(compute_done) != cudaSuccess)
            continue;

        if (attempts.load() > 0)
        {
            cudaStreamWaitEvent(transfer_stream, compute_done);

            cudaError_t copyError = cudaMemcpyAsync(h_results, d_results,
                                                    actual_batch_size * sizeof(CurvePoint),
                                                    cudaMemcpyDeviceToHost, transfer_stream);
            if (copyError == cudaSuccess)
            {
                copyError = cudaMemcpyAsync(h_scalars, d_scalars,
                                            actual_batch_size * sizeof(Scalar),
                                            cudaMemcpyDeviceToHost, transfer_stream);
            }

            if (copyError != cudaSuccess)
            {
                std::cerr << "Memory transfer failed: " << cudaGetErrorString(copyError) << "\n";
                continue;
            }

            cudaEventRecord(transfer_done, transfer_stream);
            if (cudaEventSynchronize(transfer_done) == cudaSuccess)
            {
                process_gpu_results(h_results, h_scalars, actual_batch_size);
            }
        }

        attempts += actual_batch_size;
        if (attempts.load() % (actual_batch_size * 10) == 0)
        {
            report_status();
        }
    }

    cudaFreeHost(h_results);
    cudaFreeHost(h_scalars);
    cudaDeviceSynchronize();
    report_status(true);
}

void PubkeySubtractor::process_gpu_results(CurvePoint *points, Scalar *scalars, size_t valid_results)
{
    for (size_t i = 0; i < valid_results; i++)
    {
        std::stringstream ss;
        ss << std::hex << std::setfill('0');

        Field y_norm;
        Field::from_montgomery(y_norm, points[i].y());
        bool is_odd = (y_norm.array()[0] & 1) != 0;
        ss << (is_odd ? "03" : "02");

        Field x_norm;
        Field::from_montgomery(x_norm, points[i].x());
        for (size_t j = Field::LimbN; j > 0; --j)
        {
            ss << std::setw(16) << x_norm.array()[j - 1];
        }
        generated_pubkey = ss.str();

        ss.str("");
        ss.clear();
        for (int j = 3; j >= 0; --j)
        {
            ss << std::hex << std::setfill('0') << std::setw(16) << scalars[i].array()[j];
        }
        current_subtraction_value = ss.str();

        if (check_bloom_filters(generated_pubkey))
        {
            save_match(generated_pubkey, scalars[i]);
        }
    }
}

void PubkeySubtractor::save_match(const std::string &pubkey, const Scalar &scalar)
{
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cout << "\nPotential match found!\n"
              << "Public key: " << pubkey << "\n"
              << "Subtraction value: ";

    for (int i = 3; i >= 0; --i)
    {
        std::cout << std::hex << std::setfill('0') << std::setw(16) << scalar.array()[i];
    }
    std::cout << std::dec << std::endl;

    std::ofstream match_file("matches.txt", std::ios::app);
    if (match_file)
    {
        match_file << "Public Key: " << pubkey << "\n"
                   << "Subtraction: ";
        for (int i = 3; i >= 0; --i)
        {
            match_file << std::hex << std::setfill('0') << std::setw(16) << scalar.array()[i];
        }
        match_file << std::dec << "\n\n";
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
        std::cout << "\rAttempts: " << current_attempts
                  << " Current Public Key: " << generated_pubkey
                  << " Current Subtraction: " << current_subtraction_value
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
    return CurvePoint::lift_x(h_input_point, mont_x, is_odd, gec_rng);
}
