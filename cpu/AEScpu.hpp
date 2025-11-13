#pragma once
#include <cstddef>
#include <cstdint>

bool aes_ctr_cpu(const uint8_t* in, uint8_t* out, size_t n,
                 const uint8_t* key, int keyBits,
                 const uint8_t iv[16]);
