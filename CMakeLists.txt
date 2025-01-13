cmake_minimum_required(VERSION 3.16)
project(btc_keys)

set(GEC_ENABLE_CUDA OFF)
set(GEC_ENABLE_AVX2 ON)
set(GEC_ENABLE_PTHREADS ON)  # Changed to ON since we need pthreads

# Find Threads package
find_package(Threads REQUIRED)

add_subdirectory(gec)

# Add xxhash library as C code
add_library(xxhash STATIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/bloom/xxhash.c
)

# Ensure xxhash is compiled as C
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/bloom/xxhash.c
    PROPERTIES LANGUAGE C
)

target_include_directories(xxhash PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/bloom
)

# Add bloom library
add_library(bloom 
    ${CMAKE_CURRENT_SOURCE_DIR}/bloom/bloom.cpp
)

target_include_directories(bloom PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/bloom
)

target_link_libraries(bloom PUBLIC xxhash)

# Main executable
add_executable(btc_keys btc_keys.cpp)

target_include_directories(btc_keys PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/gec/include
    ${CMAKE_CURRENT_SOURCE_DIR}/bloom
)

target_link_libraries(btc_keys PRIVATE 
    gec 
    secp256k1
    bloom
    xxhash
    Threads::Threads  # Added pthread support
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT MSVC)
    target_compile_options(btc_keys PRIVATE -O3)
    if(GEC_ENABLE_AVX2)
        target_compile_options(btc_keys PRIVATE -march=native)
    endif()
else()
    target_compile_options(btc_keys PRIVATE /O2)
    if(GEC_ENABLE_AVX2)
        target_compile_options(btc_keys PRIVATE /arch:AVX2)
    endif()
endif()
