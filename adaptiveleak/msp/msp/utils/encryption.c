#include "encryption.h"


uint16_t round_to_aes_block(uint16_t numBytes) {
    uint16_t remainder = (AES_BLOCK_SIZE - (numBytes & 0xF)) & 0xF;
    return numBytes + remainder;
}
