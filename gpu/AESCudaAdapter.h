#pragma once
#include <cstdint>
#include <cstddef>

namespace AESCuda {

#ifndef __CUDACC__   // If CUDA compiler is NOT active
inline bool isAvailable() { return false; }

inline bool ctrEncrypt(const uint8_t*, uint8_t*, size_t,
                       const uint8_t*, int,
                       const uint8_t[16])
{ return false; }
#endif

}
