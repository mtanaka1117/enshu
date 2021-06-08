#include <stdint.h>

#include "constants.h"
#include "bitmap.h"
#include "fixed_point.h"

#ifndef ENCODING_H_
#define ENCODING_H_

#define TWO_BYTE_OFFSET 32767

uint16_t encode_standard(uint8_t *output, FixedPoint *features, struct BitMap *collectedIndices, uint16_t numFeatures);

#endif
