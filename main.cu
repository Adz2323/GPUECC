#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <csignal>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "subtractor.cuh"

// Helper function to convert SM version to cores
int _ConvertSMVer2Cores(int major, int minor)
{
    struct SMtoCores
    {
        int SM;
        int Cores;
    };

    SMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128}, {0x52, 128}, {0x53, 128}, {0x60, 64}, {0x61, 128}, {0x62, 128}, {0x70, 64}, {0x72, 64}, {0x75, 64}, {0x80, 64}, {0x86, 128}, {0x89, 128}, {0x90, 128}, {-1, -1}};

    for (int i = 0; nGpuArchCoresPerSM[i].SM != -1; i++)
    {
        if (nGpuArchCoresPerSM[i].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[i].Cores;
        }
    }
    return 64;
}

std::atomic<bool> g_should_stop{false};

void signal_handler(int)
{
    g_should_stop = true;
    std::cout << "\nReceived interrupt signal. Shutting down gracefully...\n";
}

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

        // Initialize CUDA
        int deviceCount;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error != cudaSuccess || deviceCount == 0)
        {
            std::cerr << "No CUDA devices available!" << std::endl;
            return 1;
        }

        // Get device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);

        const std::string pubkey = argv[1];
        const std::string range_str = argv[2];
        const std::string bloom_file = argv[4];

        size_t colon_pos = range_str.find(':');
        if (colon_pos == std::string::npos)
        {
            std::cerr << "Invalid range format. Use start:end" << std::endl;
            return 1;
        }

        // Setup signal handler
        signal(SIGINT, signal_handler);

        // Set optimal batch size
        const size_t optimal_batch_size = deviceProp.maxThreadsPerMultiProcessor *
                                          deviceProp.multiProcessorCount;

        PubkeySubtractor subtractor(pubkey, optimal_batch_size);

        std::cout << "Loading bloom filters from: " << bloom_file << std::endl;
        if (!subtractor.init_bloom_filters(bloom_file))
        {
            std::cerr << "Failed to initialize bloom filters" << std::endl;
            return 1;
        }

        // Parse range values
        auto [start_scalar, end_scalar] = subtractor.parse_range(
            range_str.substr(0, colon_pos),
            range_str.substr(colon_pos + 1));

        std::cout << "\nConfiguration:\n"
                  << "Public Key: " << pubkey << "\n"
                  << "Range Start: " << range_str.substr(0, colon_pos) << "\n"
                  << "Range End: " << range_str.substr(colon_pos + 1) << "\n"
                  << "Bloom Filter File: " << bloom_file << "\n"
                  << "GPU Device: " << deviceProp.name << "\n"
                  << "CUDA Cores: " << deviceProp.multiProcessorCount * _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) << "\n"
                  << "Batch Size: " << optimal_batch_size << "\n\n";

        subtractor.subtract_range(start_scalar, end_scalar, g_should_stop);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\nProgram completed successfully." << std::endl;
    return 0;
}
