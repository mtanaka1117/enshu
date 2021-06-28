#include "range_shifting.h"

#ifdef IS_MSP
#pragma DATA_SECTION("UNION_FIND", ".matrix")
#endif
static struct ShiftGroup UNION_FIND[100];

static uint8_t SCORE_BUFFER[100];
static uint8_t LEFT_PARENTS[100];


int8_t get_range_shift(FixedPoint value, uint8_t currentPrecision, uint8_t newWidth, uint8_t numShiftBits, int8_t prevShift) {
    if (newWidth == 16) {
        return 0;
    }

    const int8_t nonFractional = 16 - currentPrecision;
    const uint8_t shiftOffset = (1 << (numShiftBits - 1)) + 1;
    const uint16_t widthMask = (1 << (newWidth - 1)) - 1;  // Mask out all non-data bits (including the sign bit)
    const uint16_t signBit = 1 << newWidth;
    const int8_t newPrecision = newWidth - nonFractional;
    const uint16_t absValue = fp_abs(value);


    volatile int8_t shift;
    volatile int8_t shiftedPrecision;
    volatile int8_t conversionShift;

    volatile FixedPoint shiftedValue;
    volatile FixedPoint error;

    // Try the previous shift value first
    shiftedPrecision = newPrecision - prevShift;
    conversionShift = currentPrecision - shiftedPrecision;

    if (conversionShift > 0) {
        shiftedValue = (absValue >> conversionShift) & widthMask;
        shiftedValue = shiftedValue << conversionShift;
    } else {
        conversionShift *= -1;
        shiftedValue = (absValue << conversionShift) & widthMask;
        shiftedValue = shiftedValue >> conversionShift;
    }

    shiftedValue &= CONV_MASK;  // Prevent negative values
    const FixedPoint prevError = fp_abs(fp_sub(absValue, shiftedValue));  // Error in the current precision

    if (prevError <= SHIFT_TOL) {
        return prevShift;
    }

    volatile FixedPoint bestError = prevError;
    volatile int8_t bestShift = prevShift;

    uint16_t i = 1 << numShiftBits;
    for (; i > 0; i--) {
        shift = i - shiftOffset;

        if (shift == prevShift) {
            continue;
        }

        shiftedPrecision = newPrecision - shift;
        conversionShift = currentPrecision - shiftedPrecision;

        if (conversionShift > 0) {
            shiftedValue = (absValue >> conversionShift) & widthMask;
            shiftedValue = shiftedValue << conversionShift;
        } else {
            conversionShift *= -1;
            shiftedValue = (absValue << conversionShift) & widthMask;
            shiftedValue = shiftedValue >> conversionShift;
        }

        shiftedValue &= CONV_MASK;  // Prevent negative values
        error = fp_sub(absValue, shiftedValue);

        if (error >= 0) {
            error = fp_abs(error);

            if (error <= bestError) {
                bestError = error;
                bestShift = shift;
            }
        }
    }

    if (prevError <= bestError) {
        return prevShift;
    }

    return bestShift;
}


void get_range_shifts_array(int8_t *result, FixedPoint *values, uint8_t currentPrecision, uint8_t newWidth, uint8_t numShiftBits, uint16_t numValues) {
    volatile int8_t prevShift = 0;

    uint16_t i;
    for (i = 0; i < numValues; i++) {
        prevShift = get_range_shift(values[i], currentPrecision, newWidth, numShiftBits, prevShift);
        result[i] = prevShift;
    }
}


uint16_t run_length_encode_shifts(int8_t *resultShifts, uint16_t *resultCounts, int8_t *shifts, uint16_t numValues) {
    volatile uint16_t resultIdx = 0;
    volatile int8_t currentShift = shifts[0];
    volatile uint16_t currentCount = 1;

    volatile int8_t tempShift;

    uint16_t i;
    for (i = 1; i < numValues; i++) {
        tempShift = shifts[i];
        if (tempShift != currentShift) {
            resultShifts[resultIdx] = currentShift;
            resultCounts[resultIdx] = currentCount;

            currentShift = tempShift;
            currentCount = 1;
            resultIdx++;
        } else {
            currentCount++;
        }
    }

    // Write the last element
    resultShifts[resultIdx] = currentShift;
    resultCounts[resultIdx] = currentCount;

    return resultIdx + 1;
}


/*
 * Shift Group Functions
 */
uint16_t find(struct ShiftGroup *group, struct ShiftGroup *unionFind) {
    int16_t parent = group->parent;

    while (parent != -1) {
        group = unionFind + parent;
        parent = group->parent;
    }

    return group->idx;
}


void merge(struct ShiftGroup *g1, struct ShiftGroup *g2, struct ShiftGroup *unionFind) {
    uint16_t idx1 = find(g1, unionFind);
    uint16_t idx2 = find(g2, unionFind);

    struct ShiftGroup *left;
    struct ShiftGroup *right;

    if (idx1 < idx2) {
        left = unionFind + idx1;
        right = unionFind + idx2;
    } else {
        left = unionFind + idx2;
        right = unionFind + idx1;
    }

    right->parent = left->idx;
    left->shift = max8(left->shift, right->shift);
    left->nextParent = right->nextParent;
    left->count = left->count + right->count;
}


void get_groups_to_merge(uint8_t *leftParents, uint8_t *scoreBuffer, uint8_t numToMerge, struct ShiftGroup *unionFind, uint16_t numGroups) {
    // Locate the first parent
    uint16_t i = 0;
    while (unionFind[i].parent != -1) {
        i++;
    }

    // Get the initial left and right groups
    volatile uint8_t idx = 0;
    volatile uint8_t finalIdx, lowerIdx;
    volatile uint8_t temp;

    volatile struct ShiftGroup *left = unionFind;
    volatile struct ShiftGroup *right;
    
    volatile int8_t score;
    volatile int8_t shiftDiff;
    volatile uint8_t nextParent = left->nextParent;

    while (nextParent < numGroups) {
        left = unionFind + idx;
        right = unionFind + nextParent;

        shiftDiff = fp_abs(left->shift - right->shift);
        score = (left->count + right->count + shiftDiff) * (shiftDiff > 0);

        finalIdx = idx - 1;
        if ((idx < numToMerge) || (score < scoreBuffer[finalIdx])) {

            if (idx < numToMerge) {
                finalIdx++;
            }

            scoreBuffer[finalIdx] = score;
            leftParents[finalIdx] = idx;

            lowerIdx = finalIdx - 1;
            while ((finalIdx > 0) && (scoreBuffer[finalIdx] < scoreBuffer[lowerIdx])) {
                temp = scoreBuffer[lowerIdx];
                scoreBuffer[lowerIdx] = scoreBuffer[finalIdx];
                scoreBuffer[finalIdx] = temp;

                temp = leftParents[lowerIdx];
                leftParents[lowerIdx] = leftParents[finalIdx];
                leftParents[finalIdx] = temp;

                finalIdx--;
                lowerIdx--;
            }
        }

        idx = nextParent;
        nextParent = unionFind[idx].nextParent;
    }
}


uint16_t create_shift_groups(int8_t *resultShifts, uint16_t *resultCounts, int8_t *shifts, uint16_t numShifts, uint16_t maxNumGroups) {
    // Create the initial groups via RLE
    uint8_t groupCount = run_length_encode_shifts(resultShifts, resultCounts, shifts, numShifts);

    if (groupCount <= maxNumGroups) {
        return groupCount;
    }

    // Initialize the union-find structure
    struct ShiftGroup *unionFind = UNION_FIND;
    int16_t countSum, shiftDiff;

    uint8_t i, j;
    for (i = 0; i < groupCount; i++) {
        j = i + 1;
        unionFind[i].idx = i;
        unionFind[i].count = resultCounts[i];
        unionFind[i].shift = resultShifts[i];
        unionFind[i].parent = -1;
        unionFind[i].nextParent = j;
    }

    // Merge the groups until we meet the right amount
    uint8_t leftIdx, rightIdx;
    uint8_t numToMerge = groupCount - maxNumGroups;

    get_groups_to_merge(LEFT_PARENTS, SCORE_BUFFER, numToMerge, unionFind, groupCount);

    for (i = 0; i < numToMerge; i++) {
        leftIdx = LEFT_PARENTS[i];
        rightIdx = unionFind[leftIdx].nextParent;
        merge(unionFind + leftIdx, unionFind + rightIdx, unionFind);
    }
    
    //while (numGroups > maxNumGroups) {
    //    get_groups_to_merge(&leftIdx, &rightIdx, unionFind, groupCount);
    //    merge(unionFind + leftIdx, unionFind + rightIdx, unionFind, groupCount);
    //    numGroups--;
    //}

    // Reconstruct the resulting shifts and counts
    uint16_t groupIdx = 0;

    for (i = 0; i < groupCount; i++) {
        if (unionFind[i].parent == -1) {
            resultShifts[groupIdx] = unionFind[i].shift;
            resultCounts[groupIdx] = unionFind[i].count;
            groupIdx++;
        }
    }

    return groupIdx;
}
