#include "AEScpu.hpp"
#include <openssl/evp.h>

bool aes_ctr_cpu(const uint8_t* in, uint8_t* out, size_t n,
                 const uint8_t* key, int keyBits,
                 const uint8_t iv[16])
{
    const EVP_CIPHER* cipher = (keyBits == 256) ? EVP_aes_256_ctr()
                           : (keyBits == 192) ? EVP_aes_192_ctr()
                                              : EVP_aes_128_ctr();

    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    int outlen;

    EVP_EncryptInit_ex(ctx, cipher, nullptr, key, iv);
    EVP_EncryptUpdate(ctx, out, &outlen, in, (int)n);
    EVP_EncryptFinal_ex(ctx, out + outlen, &outlen);
    EVP_CIPHER_CTX_free(ctx);
    return true;
}
