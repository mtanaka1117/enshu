#include <stdint.h>
#include "fixed_point.h"
#include "bitmap.h"
#include "matrix.h"


#ifndef PRUNING_H_
#define PRUNING_H_

void prune_sequence(struct Vector *measurements, struct BitMap *collectedIndices, uint16_t numCollected, uint16_t maxCollected, uint16_t seqLength, uint16_t precision);

#endif
