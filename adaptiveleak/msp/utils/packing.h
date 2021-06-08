#include <stdint.h>

#include "fixed_point.h"

#ifndef PACKING_H_
#define PACKING_H_

uint16_t *pack(uint8_t *output, struct FixedPoint *values, uint8_t bitWidth, uint16_t numValues);
uint16_t packed_length(uint16_t numValues, uint8_t bitWidth);

#endif
