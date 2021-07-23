#include "encryption.h"


uint16_t round_to_aes_block(uint16_t numBytes) {
    uint16_t remainder = (AES_BLOCK_SIZE - (numBytes & 0xF)) & 0xF;
    return numBytes + remainder;
}


#ifdef IS_MSP
void encrypt_aes128(uint8_t *data, const uint8_t *prev, uint8_t *outputBuffer, uint16_t numBytes) {
    uint16_t i;
    for (i = 0; i < numBytes; i += AES_BLOCK_SIZE) {
        AES256_encryptData(AES256_BASE, data + i, prev, outputBuffer + i);
        prev = outputBuffer + i;
    }
}
#endif
