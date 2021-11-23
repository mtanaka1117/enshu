#include <stdint.h>

#ifdef IS_MSP
#include <msp430.h>
#include "aes256.h"
#endif

#ifndef ENCRYPTION_H_
#define ENCRYPTION_H_

#define AES_BLOCK_SIZE 16
#define CHACHA_NONCE_LEN 12

uint16_t round_to_aes_block(uint16_t numBytes);

#ifdef IS_MSP
void encrypt_aes128(uint8_t *data, const uint8_t *prev, uint8_t *outputBuffer, uint16_t numBytes);
#endif

#endif
