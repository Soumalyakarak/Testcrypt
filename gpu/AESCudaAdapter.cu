#include "AESCudaAdapter.h"
#include <cuda_runtime.h>
#include <cstring>

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

    // Allocate device buffers
    uint8_t *dIn, *dOut;
    cudaMalloc(&dIn, nBytes);
    cudaMalloc(&dOut, nBytes);
    cudaMemcpy(dIn, inHost, nBytes, cudaMemcpyHostToDevice);

    // Call the AES kernel from whichever repo you chose
    // Example signature from Tezcan’s repo:
    // aes_encrypt_ctr_gpu(dOut, dIn, key, iv, nBlocks);

    size_t nBlocks = (nBytes + 15) / 16;
    aes_encrypt_ctr_gpu<<<(nBlocks+255)/256,256>>>(dOut, dIn, key, iv, nBlocks);

    cudaMemcpy(outHost, dOut, nBytes, cudaMemcpyDeviceToHost);
    cudaFree(dIn);
    cudaFree(dOut);
    cudaDeviceSynchronize();
    return true;
}

} // namespace AESCuda
