#include <stdint.h>
#include "utils/fixed_point.h"
#include "utils/constants.h"
#include "data.h"
#include "policy_parameters.h"

#ifndef SAMPLER_H_
#define SAMPLER_H_
uint8_t get_measurement(FixedPoint *result, uint16_t seqNum, uint16_t elemNum, uint16_t numFeatures, uint16_t seqLength);
#endif
