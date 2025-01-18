#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <csignal>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iomanip>
#include "subtractor.cuh"

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
                      << " " << RANGE_START << ":" << RANGE_END
                      << " -f scanned_pubkeys.bin" << std::endl;
            return 1;
        }

        std::cout << "Initializing CUDA...\n";
        int deviceCount;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);
        if (error != cudaSuccess || deviceCount == 0)
        {
            std::cerr << "No CUDA devices available!" << std::endl;
            return 1;
        }
        std::cout << "Found " << deviceCount << " CUDA device(s)\n";

        cudaDeviceProp deviceProp;
        error = cudaGetDeviceProperties(&deviceProp, 0);
        if (error != cudaSuccess)
        {
            std::cerr << "Failed to get device properties: " << cudaGetErrorString(error) << std::endl;
            return 1;
        }

        const std::string pubkey = argv[1];
        const std::string bloom_file = argv[4];

        std::cout << "Range Confirmation:\n"
                  << "Start (Hex): 0x" << RANGE_START << "\n"
                  << "End (Hex): 0x" << RANGE_END << "\n\n"
                  << "Proceed? (Y/N): ";

        char response;
        std::cin >> response;
        if (response != 'Y' && response != 'y')
        {
            std::cout << "Operation cancelled.\n";
            return 0;
        }

        signal(SIGINT, signal_handler);
        std::cout << "Signal handler initialized\n";

        const size_t optimal_batch_size = deviceProp.maxThreadsPerMultiProcessor *
                                          deviceProp.multiProcessorCount;

        std::cout << "Creating PubkeySubtractor with batch size: " << optimal_batch_size << "\n";
        PubkeySubtractor subtractor(pubkey, optimal_batch_size);
        std::cout << "Loading bloom filters from: " << bloom_file << std::endl;

        if (!subtractor.init_bloom_filters(bloom_file))
        {
            std::cerr << "Failed to initialize bloom filters" << std::endl;
            return 1;
        }

        std::cout << "Parsing hex ranges...\n";
        auto start_scalar = parse_hex_string(RANGE_START);
        auto end_scalar = parse_hex_string(RANGE_END);

        // Debug output
        std::cout << "Start scalar (hex): ";
        for (int i = 3; i >= 0; i--)
        {
            std::cout << std::hex << std::setw(16) << std::setfill('0')
                      << start_scalar.array()[i] << " ";
        }
        std::cout << "\nEnd scalar (hex): ";
        for (int i = 3; i >= 0; i--)
        {
            std::cout << std::hex << std::setw(16) << std::setfill('0')
                      << end_scalar.array()[i] << " ";
        }
        std::cout << std::dec << std::endl;

        std::cout << "\nConfiguration:\n"
                  << "Public Key: " << pubkey << "\n"
                  << "Range Start: 0x" << RANGE_START << "\n"
                  << "Range End: 0x" << RANGE_END << "\n"
                  << "Bloom Filter File: " << bloom_file << "\n"
                  << "GPU Device: " << deviceProp.name << "\n"
                  << "CUDA Cores: " << deviceProp.multiProcessorCount * _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) << "\n"
                  << "Batch Size: " << optimal_batch_size << "\n\n";

        std::cout << "Starting subtraction process...\n";
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
