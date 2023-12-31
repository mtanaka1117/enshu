#include <stdint.h>

#include "constants.h"
#include "bitmap.h"
#include "fixed_point.h"
#include "matrix.h"
#include "encryption.h"
#include "packing.h"
#include "pruning.h"
#include "range_shifting.h"
#include "math.h"

#ifndef ENCODING_H_
#define ENCODING_H_

#define TWO_BYTE_OFFSET 32768u

uint16_t encode_standard(uint8_t *output, struct Vector *features, struct BitMap *collectedIndices, uint16_t numFeatures, uint16_t seqLength);

uint16_t encode_group(uint8_t *output, struct Vector *featureVectors, struct BitMap *collectedIndices, uint16_t numCollected, uint16_t numFeatures, uint16_t seqLength, uint16_t targetBytes, uint16_t precision, uint16_t maxCollected, FixedPoint *tempBuffer, int8_t *shiftBuffer, uint16_t *countBuffer, uint8_t isBlock);
void set_group_widths(uint8_t *result, uint16_t *groupSizes, uint8_t numGroups, uint16_t targetBytes, uint16_t startWidth);
uint16_t get_max_num_groups(const uint16_t targetBytes, const uint16_t numCollected, const uint16_t numFeatures, const uint16_t width);
uint16_t calculate_grouped_size(uint8_t *groupWidths, uint16_t numCollected, uint16_t numFeatures, uint16_t seqLength, uint16_t groupSize, uint16_t numGroups, uint8_t isBlock);

#endif
