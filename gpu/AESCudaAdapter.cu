#include "AESCudaAdapter.h"

#include <cuda_runtime.h>
#include <cstring>

// Include Tezcan CUDA AES kernels
#include "AES_final.h"
#include "256-ctr.cuh"

namespace AESCuda {

bool isAvailable() {
    int count = 0;
    return (cudaGetDeviceCount(&count) == cudaSuccess && count > 0);
}

bool ctrEncrypt(const uint8_t* inHost, uint8_t* outHost, size_t nBytes,
                const uint8_t* key, int keyBits,
                const uint8_t iv[16])
{
    if (!isAvailable() || nBytes == 0) return false;

    // AES block size
    size_t nBlocks = (nBytes + 15) / 16;

    // Device buffers
    uint8_t *dIn, *dOut, *dIV, *dKey;
    cudaMalloc(&dIn, nBytes);
    cudaMalloc(&dOut, nBytes);
    cudaMalloc(&dIV, 16);
    cudaMalloc(&dKey, 32);   // AES-256

    cudaMemcpy(dIn,  inHost, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dIV,  iv,     16,     cudaMemcpyHostToDevice);
    cudaMemcpy(dKey, key,    32,     cudaMemcpyHostToDevice);

    // Launch Tezcan AES-256 CTR kernel
    aes256_ctr_encrypt_gpu<<<(nBlocks + 255) / 256, 256>>>(
        dOut,
        dIn,
        dKey,
        dIV,
        nBlocks,
        nBytes
    );

    cudaDeviceSynchronize();
    cudaMemcpy(outHost, dOut, nBytes, cudaMemcpyDeviceToHost);

    cudaFree(dIn);
    cudaFree(dOut);
    cudaFree(dIV);
    cudaFree(dKey);

    return true;
}

} // namespace AESCuda
