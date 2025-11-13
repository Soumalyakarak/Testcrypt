#include "Cryption.hpp"
#include "cpu/AEScpu.hpp"
#include "gpu/AESCudaAdapter.h"
#include <stdexcept>

namespace Cryption {

bool gpuAvailable() {
    return AESCuda::isAvailable();
}

void aesCtrBuffer(const uint8_t* in, uint8_t* out, size_t n,
                  const uint8_t key[32], int keyBits,
                  const uint8_t iv[16], bool useGPU)
{
    if (useGPU && AESCuda::isAvailable()) {
        if (!AESCuda::ctrEncrypt(in, out, n, key, keyBits, iv))
            throw std::runtime_error("CUDA AES failed");
    } else {
        if (!aes_ctr_cpu(in, out, n, key, keyBits, iv))
            throw std::runtime_error("CPU AES failed");
    }
}

// not used yet
void encryptFileCTR(...) {}
void decryptFileCTR(...) {}

}
