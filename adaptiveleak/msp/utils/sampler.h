#include <stdint.h>
#include "../data.h"
#include "fixed_point.h"

#ifndef SAMPLER_H_
#define SAMPLER_H_

    uint8_t get_measurement(FixedPoint *result, uint16_t seqNum, uint16_t elemNum, uint16_t numFeatures, uint16_t seqLength);

#endif
