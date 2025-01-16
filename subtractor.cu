#include "subtractor.cuh"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <thread>

namespace cg = cooperative_groups;

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
    if (bloom_init2(&h_bloom_filter1, MAX_ENTRIES1, BLOOM1_FP_RATE) != 0 ||
        bloom_init2(&h_bloom_filter2, MAX_ENTRIES2, BLOOM2_FP_RATE) != 0 ||
        bloom_init2(&h_bloom_filter3, MAX_ENTRIES3, BLOOM3_FP_RATE) != 0)
    {
        return false;
    }
    bloom_initialized = true;

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
    std::mutex file_mutex;
    std::atomic<size_t> entries_processed{0};
    std::vector<WorkerBuffer> buffers(num_threads);

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
                printf("\rProcessed %zu/%zu entries (%.2f entries/sec)...",
                       entries_processed.load(), total_entries, rate);
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

    printf("\nCompleted loading %zu entries\n", entries_processed.load());
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
    const size_t block_size = 256;
    const size_t num_blocks = (batch_size + block_size - 1) / block_size;

    std::vector<CurvePoint> h_results(batch_size);
    std::vector<Scalar> h_scalars(batch_size);

    // Calculate step size for batch processing
    Scalar range = end;
    range.sub(range, start);
    Scalar step;

    // Initialize step to 1
    for (int i = 0; i < 4; i++)
    {
        step.array()[i] = 0;
    }
    step.array()[0] = 1;

    // Divide range by batch_size * 100 using repeated subtraction
    Scalar temp = range;
    uint64_t divisor = batch_size * 100;
    uint64_t count = 0;
    while (temp >= step)
    {
        temp.sub(temp, step);
        count++;
    }

    // Set step to calculated value
    for (int i = 0; i < 4; i++)
    {
        step.array()[i] = 0;
    }
    step.array()[0] = count / divisor;

    Scalar current = start;
    void *args[] = {&d_input_point, &current, &step, &batch_size, &d_results, &d_scalars};

    while (!should_stop && current < end)
    {
        // Launch kernel asynchronously
        cudaLaunchCooperativeKernel(
            (void *)subtract_kernel,
            num_blocks, block_size,
            args, 0, compute_stream);
        cudaEventRecord(compute_done, compute_stream);

        // Overlap computation with previous batch processing
        if (attempts.load() > 0)
        {
            cudaStreamWaitEvent(transfer_stream, compute_done);

            // Transfer results asynchronously
            cudaMemcpyAsync(h_results.data(), d_results,
                            batch_size * sizeof(CurvePoint),
                            cudaMemcpyDeviceToHost,
                            transfer_stream);
            cudaMemcpyAsync(h_scalars.data(), d_scalars,
                            batch_size * sizeof(Scalar),
                            cudaMemcpyDeviceToHost,
                            transfer_stream);
            cudaEventRecord(transfer_done, transfer_stream);

            // Process results while next batch computes
            cudaEventSynchronize(transfer_done);
            process_gpu_results(h_results, h_scalars, batch_size);
        }

        // Move to next batch
        for (size_t i = 0; i < batch_size; i++)
        {
            current.add(current, step);
        }
        attempts += batch_size;

        // Report progress periodically
        if (attempts.load() % (batch_size * 10) == 0)
        {
            report_status();
        }
    }

    // Process final batch
    cudaEventSynchronize(compute_done);
    cudaMemcpy(h_results.data(), d_results, batch_size * sizeof(CurvePoint), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_scalars.data(), d_scalars, batch_size * sizeof(Scalar), cudaMemcpyDeviceToHost);
    process_gpu_results(h_results, h_scalars, batch_size);

    report_status(true);
}

void PubkeySubtractor::process_gpu_results(
    const std::vector<CurvePoint> &points,
    const std::vector<Scalar> &scalars,
    size_t valid_results)
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
        std::string compressed = ss.str();

        if (check_bloom_filters(compressed))
        {
            save_match(compressed, scalars[i]);
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
        std::cout << "\rProcessed " << current_attempts << " keys (" << std::fixed
                  << std::setprecision(2) << kps << " k/s)" << std::flush;
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

std::pair<PubkeySubtractor::Scalar, PubkeySubtractor::Scalar>
PubkeySubtractor::parse_range(const std::string &start_str, const std::string &end_str)
{
    Scalar start, end;

    // Initialize scalars
    for (int i = 0; i < 4; i++)
    {
        start.array()[i] = 0;
        end.array()[i] = 0;
    }

    try
    {
        // Parse start value
        size_t pos = 0;
        int limb = 0;
        while (pos < start_str.length() && limb < 4)
        {
            size_t chunk_size = std::min<size_t>(16, start_str.length() - pos);
            std::string chunk = start_str.substr(pos, chunk_size);
            start.array()[limb++] = std::stoull(chunk);
            pos += chunk_size;
        }

        // Parse end value
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
        throw std::runtime_error("Error parsing range values: " + std::string(e.what()));
    }

    return {start, end};
}
