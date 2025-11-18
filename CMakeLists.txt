cmake_minimum_required(VERSION 3.18)
project(Testcrypt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenSSL REQUIRED)
include_directories(${OPENSSL_INCLUDE_DIR})

# Base sources
add_library(cryption
    Cryption.cpp
    cpu/AEScpu.cpp
)

# Detect CUDA Toolkit
find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    message(STATUS "CUDA Found Enabling GPU Mode")

    # Add CUDA kernel files
    target_sources(cryption PRIVATE
        gpu/AESCudaAdapter.cu
        gpu/AES_final.cu         
    )

    target_include_directories(cryption PUBLIC gpu)
    target_link_libraries(cryption PUBLIC CUDA::cudart)

else()
    message(STATUS "CUDA Not Found ! CPU Mode Only")
endif()

target_include_directories(cryption PUBLIC . cpu gpu)
target_link_libraries(cryption PUBLIC OpenSSL::SSL OpenSSL::Crypto)

add_executable(testAES testAES.cpp)
target_link_libraries(testAES PRIVATE cryption)
