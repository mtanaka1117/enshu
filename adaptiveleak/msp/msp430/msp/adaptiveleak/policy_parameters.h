#include <stdint.h>
#include "utils/matrix.h"

#ifndef POLICY_PARAMETERS_H_
#define POLICY_PARAMETERS_H_
#define BITMASK_BYTES 7
#define SEQ_LENGTH 50
#define NUM_FEATURES 6
#define DEFAULT_WIDTH 16
#define DEFAULT_PRECISION 13
#define TARGET_BYTES 336

#define IS_STANDARD_ENCODED
#define IS_UNIFORM
#define NUM_INDICES 25
static const uint16_t COLLECT_INDICES[NUM_INDICES] = {0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48};
#endif
