#include <stdint.h>

#ifndef LFSR_H_
#define LFSR_H_

uint16_t lfsr(uint16_t start, uint16_t steps);
void lfsr_array(uint8_t *startValues, const uint16_t *steps, uint8_t numValues);

#endif
