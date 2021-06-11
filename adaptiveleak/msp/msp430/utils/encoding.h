#include <stdint.h>

#include "constants.h"
#include "bitmap.h"
#include "fixed_point.h"
#include "matrix.h"

#ifndef ENCODING_H_
#define ENCODING_H_

#define TWO_BYTE_OFFSET 32768u

uint16_t encode_standard(uint8_t *output, struct Vector *features, struct BitMap *collectedIndices, uint16_t numFeatures, uint16_t seqLength);

#endif
