#include <stdint.h>
#include "utils/matrix.h"

#ifndef POLICY_PARAMETERS_H_
#define POLICY_PARAMETERS_H_
#define BITMASK_BYTES 3
#define SEQ_LENGTH 23
#define NUM_FEATURES 10
#define DEFAULT_WIDTH 16
#define DEFAULT_PRECISION 0
#define TARGET_BYTES 178
#define TARGET_DATA_BYTES 160

#define IS_GROUP_ENCODED
#define MAX_COLLECTED 22
#define IS_ADAPTIVE_DEVIATION
#define DEVIATION_PRECISION 5
#define ALPHA 22
#define BETA 22
#define THRESHOLD 6407
#define MAX_SKIP 2
#define MIN_SKIP 1
#endif
