#include <stdint.h>
#include "fixed_point.h"
#include "math.h"

#ifndef RANGE_SHIFTING_H_
#define RANGE_SHIFTING_H_

    #define CONV_MASK 0x7FFF
    #define SHIFT_TOL 1

    struct ShiftGroup {
        uint8_t idx;
        int8_t parent;
        int8_t shift;
        int8_t nextParent;
        uint16_t count;
    };

    int8_t get_range_shift(FixedPoint value, uint8_t currentPrecision, uint8_t newWidth, uint8_t numShiftBits, int8_t prevShift);
    void get_range_shifts_array(int8_t *result, FixedPoint *values, uint8_t currentPrecision, uint8_t newWidth, uint8_t numShiftBits, uint16_t numValues);
    uint16_t run_length_encode_shifts(int8_t *resultShifts, uint16_t *resultCounts, int8_t *shifts, uint16_t numValues);

    uint16_t find(struct ShiftGroup *group, struct ShiftGroup *unionFind);
    void merge(struct ShiftGroup *g1, struct ShiftGroup *g2, struct ShiftGroup *unionFind);
    void get_groups_to_merge(uint8_t *leftParents, uint8_t *scoreBuffer, uint8_t numToMerge, struct ShiftGroup *unionFind, uint16_t numGroups);
    uint16_t create_shift_groups(int8_t *resultShifts, uint16_t *resultCounts, int8_t *shifts, uint16_t numShifts, uint16_t maxNumGroups);

#endif
