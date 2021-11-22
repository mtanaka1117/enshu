#include <stdint.h>
#include "utils/matrix.h"

#ifndef POLICY_PARAMETERS_H_
#define POLICY_PARAMETERS_H_
#define IS_MSP
#define BITMASK_BYTES 3
#define SEQ_LENGTH 23
#define NUM_FEATURES 10
#define DEFAULT_WIDTH 16
#define DEFAULT_PRECISION 0
#define TARGET_BYTES 482
#define TARGET_DATA_BYTES 464

#define IS_STANDARD_ENCODED
#define IS_PADDED
#define IS_ADAPTIVE_DEVIATION
#define DEVIATION_PRECISION 5
#define ALPHA 22
#define BETA 22
#define THRESHOLD 0
#define MAX_SKIP 1
#define MIN_SKIP 0
#endif
