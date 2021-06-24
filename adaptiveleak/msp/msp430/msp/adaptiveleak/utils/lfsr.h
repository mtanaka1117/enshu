#include <stdint.h>

#ifndef LFSR_H_
#define LFSR_H_

uint16_t lfsr(uint16_t start);
void lfsr_array(uint8_t *startValues, uint8_t numValues);

#endif
