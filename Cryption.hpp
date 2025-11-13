#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <string>

namespace Cryption {

struct KeyMaterial {
    std::vector<uint8_t> aesKey;
    std::vector<uint8_t> iv;
    std::vector<uint8_t> hmacKey; // not used now
};

struct Options {
    bool tryGPU = true;
};

bool gpuAvailable();

void aesCtrBuffer(const uint8_t* in, uint8_t* out, size_t n,
                  const uint8_t key[32], int keyBits,
                  const uint8_t iv[16], bool useGPU);

void encryptFileCTR(const std::string&, const std::string&, const KeyMaterial&, const Options&);
void decryptFileCTR(const std::string&, const std::string&, const KeyMaterial&, const Options&);

}
