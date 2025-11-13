#include <iostream>
#include <vector>
#include <cstring>
#include <random>
#include "Cryption.hpp"

static std::vector<uint8_t> randomBytes(size_t n) {
    std::vector<uint8_t> b(n);
    std::random_device rd;
    for (size_t i = 0; i < n; i++) b[i] = rd() & 0xFF;
    return b;
}

int main() {
    const char* msg = "Hello GPU AES encryption test!";
    size_t len = strlen(msg);

    std::vector<uint8_t> plain(len), cipher(len), decrypted(len);
    memcpy(plain.data(), msg, len);

    Cryption::KeyMaterial km;
    km.aesKey = randomBytes(32);
    km.iv = randomBytes(16);

    bool gpu = Cryption::gpuAvailable();
    std::cout << (gpu ? "GPU FOUND\n" : "CPU MODE(GPU not available)\n");

    Cryption::aesCtrBuffer(plain.data(), cipher.data(), len,
        km.aesKey.data(), km.aesKey.size() * 8, km.iv.data(), true);

    Cryption::aesCtrBuffer(cipher.data(), decrypted.data(), len,
        km.aesKey.data(), km.aesKey.size() * 8, km.iv.data(), true);

    if (memcmp(plain.data(), decrypted.data(), len) == 0)
        std::cout << "AES TEST PASSED\n";
    else
        std::cout << "AES TEST FAILED\n";
}
