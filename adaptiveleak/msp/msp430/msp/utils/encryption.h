#include <stdint.h>

#ifndef ENCRYPTION_H_
#define ENCRYPTION_H_

#define AES_BLOCK_SIZE 16
#define CHACHA_NONCE_LEN 12

uint16_t round_to_aes_block(uint16_t numBytes);

#endif
