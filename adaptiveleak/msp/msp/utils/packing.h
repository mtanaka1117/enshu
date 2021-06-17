#include <stdint.h>
#include "constants.h"

#ifndef PACKING_H_
#define PACKING_H_

uint16_t pack(uint8_t *output, int16_t *values, uint8_t bitWidth, uint16_t numValues, uint8_t shouldOffset);
uint16_t packed_length(uint16_t numValues, uint8_t bitWidth);

#endif
