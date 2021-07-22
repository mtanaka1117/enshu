#include <stdint.h>
#include "utils/matrix.h"

#ifndef POLICY_PARAMETERS_H_
#define POLICY_PARAMETERS_H_
#define BITMASK_BYTES 7
#define SEQ_LENGTH 50
#define NUM_FEATURES 6
#define DEFAULT_WIDTH 16
#define DEFAULT_PRECISION 13
#define TARGET_BYTES 576

#define IS_GROUP_ENCODED
#define MAX_COLLECTED 142
#define SIZE_BYTES 5

#define IS_ADAPTIVE_DEVIATION
#define ALPHA 5734
#define BETA 5734
#define THRESHOLD 33
#define MAX_SKIP 3
#endif
