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

#define IS_GROUP_ENCODED
#define IS_ADAPTIVE_HEURISTIC
#define THRESHOLD 5225
#define MAX_SKIP 3
#endif
