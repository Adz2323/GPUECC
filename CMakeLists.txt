cmake_minimum_required(VERSION 3.17)
project(btc_keys VERSION 0.1.0 DESCRIPTION "Bitcoin Key Subtractor" LANGUAGES C CXX)

option(GEC_ENABLE_CUDA "Build CUDA version" ON)
option(GEC_ENABLE_AVX2 "Enable AVX2 optimizations" ON)
option(GEC_ENABLE_PTHREADS "Enable pthreads support" ON)

# Common configurations
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

# Add xxhash library
add_library(xxhash STATIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/bloom/xxhash.c
)
target_include_directories(xxhash PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/bloom
)

# Add bloom library
add_library(bloom STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/bloom/bloom.cpp
)
target_include_directories(bloom PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/bloom
)
target_link_libraries(bloom PUBLIC xxhash)

# Add the gec library
add_subdirectory(gec)

# CPU-only version (btc_keys)
add_executable(btc_keys 
  btc_keys.cpp
)

target_include_directories(btc_keys PRIVATE 
  ${CMAKE_CURRENT_SOURCE_DIR}/gec/include
  ${CMAKE_CURRENT_SOURCE_DIR}/bloom
)

target_link_libraries(btc_keys PRIVATE 
  gec 
  secp256k1
  bloom
  xxhash
  Threads::Threads
  OpenMP::OpenMP_CXX
)

if(GEC_ENABLE_AVX2)
  if(NOT MSVC)
    target_compile_options(btc_keys PRIVATE -march=native)
  else()
    target_compile_options(btc_keys PRIVATE /arch:AVX2)
  endif()
endif()

# CUDA version (cuda_keys)
if(GEC_ENABLE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  add_library(secp256k1_cuda STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/gec/src/secp256k1.cu
  )
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/gec/src/secp256k1.cu
    PROPERTIES LANGUAGE CUDA
  )
  target_include_directories(secp256k1_cuda PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/gec/include
  )
  target_link_libraries(secp256k1_cuda PUBLIC gec)
  set_target_properties(secp256k1_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
  )

  add_executable(cuda_keys 
    main.cu
    subtractor.cu
  )

  set_source_files_properties(
    main.cu
    subtractor.cu
    PROPERTIES LANGUAGE CUDA
  )

  set_target_properties(cuda_keys PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  target_compile_options(cuda_keys PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
      --ptxas-options=-v
      --use_fast_math
      --expt-relaxed-constexpr
      --extended-lambda
      -rdc=true
    >
  )

  target_include_directories(cuda_keys PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/gec/include
    ${CMAKE_CURRENT_SOURCE_DIR}/bloom
    ${CUDAToolkit_INCLUDE_DIRS}
  )

  target_link_libraries(cuda_keys PRIVATE 
    gec 
    secp256k1_cuda
    secp256k1
    bloom
    xxhash
    Threads::Threads
    CUDA::curand
    CUDA::cudart
    OpenMP::OpenMP_CXX
  )
endif()
