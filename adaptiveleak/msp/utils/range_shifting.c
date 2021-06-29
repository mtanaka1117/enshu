#include "range_shifting.h"

//#ifdef IS_MSP
//#pragma DATA_SECTION("UNION_FIND", ".matrix")
//#endif
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

    volatile FixedPoint lastError = INT16_MAX;
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
        error = fp_abs(fp_sub(absValue, shiftedValue));

        if (error <= bestError) {
            bestError = error;
            bestShift = shift;
        } else if (error >= lastError) {
            break;
        }

        lastError = error;
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
uint8_t find(uint8_t idx, struct ShiftGroup *unionFind) {
    uint8_t prevIdx = idx;
    int16_t parent = unionFind[idx].parent;

    while (parent != -1) {
        prevIdx = parent;
        parent = unionFind[parent].parent;
    }

    return prevIdx;
}


void merge(uint8_t idx1, uint8_t idx2, struct ShiftGroup *unionFind) {
    uint8_t p1 = find(idx1, unionFind);
    uint8_t p2 = find(idx2, unionFind);

    struct ShiftGroup *left;
    struct ShiftGroup *right;

    if (p1 > p2) {
        uint8_t temp = p2;
        p2 = p1;
        p1 = temp;
    }

    left = unionFind + p1;
    right = unionFind + p2;

    right->parent = p1;
    left->shift = max8(left->shift, right->shift);
    left->count = left->count + right->count;
}


void create_union_find(struct ShiftGroup *unionFind, uint8_t *leftParents, uint8_t *scoreBuffer, int8_t *shifts, uint16_t *counts, uint8_t numToMerge, uint16_t numGroups) {
    volatile uint8_t finalIdx, lowerIdx;
    volatile uint8_t temp;

    volatile struct ShiftGroup *left;
    volatile struct ShiftGroup *right;
    
    volatile int8_t score;
    volatile int8_t shiftDiff;

    uint8_t groupIdx, nextIdx;
    for (groupIdx = 0; groupIdx < numGroups - 1; groupIdx++) {
        nextIdx = groupIdx + 1;

        shiftDiff = fp_abs(shifts[groupIdx] - shifts[nextIdx]);
        score = (counts[groupIdx] + counts[nextIdx] + shiftDiff) * (shiftDiff > 0);

        finalIdx = numToMerge - 1;
        if ((groupIdx < numToMerge) || (score < scoreBuffer[finalIdx])) {

            if (groupIdx < numToMerge) {
                finalIdx = groupIdx;
            }

            scoreBuffer[finalIdx] = score;
            leftParents[finalIdx] = groupIdx;

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

        // Initialize the union-find structure
        unionFind[groupIdx].parent = -1;
        unionFind[groupIdx].shift = shifts[groupIdx];
        unionFind[groupIdx].count = counts[groupIdx];
    }

    // Initialize the final union-find element
    groupIdx = numGroups - 1;
    unionFind[groupIdx].parent = -1;
    unionFind[groupIdx].shift = shifts[groupIdx];
    unionFind[groupIdx].count = counts[groupIdx];
}


uint16_t create_shift_groups(int8_t *resultShifts, uint16_t *resultCounts, int8_t *shifts, uint16_t numShifts, uint16_t maxNumGroups) {
    // Create the initial groups via RLE
    uint8_t groupCount = run_length_encode_shifts(resultShifts, resultCounts, shifts, numShifts);

    if (groupCount <= maxNumGroups) {
        return groupCount;
    }

    // Initialize the union-find structure and get the groups to merge
    struct ShiftGroup *unionFind = UNION_FIND;
    uint8_t leftIdx, rightIdx;
    uint8_t numToMerge = groupCount - maxNumGroups;

    create_union_find(UNION_FIND, LEFT_PARENTS, SCORE_BUFFER, resultShifts, resultCounts, numToMerge, groupCount);

    uint8_t i;
    for (i = 0; i < numToMerge; i++) {
        leftIdx = LEFT_PARENTS[i];
        merge(leftIdx, leftIdx + 1, UNION_FIND);
    }

    // Reconstruct the resulting shifts and counts
    uint16_t groupIdx = 0;

    for (i = 0; i < groupCount; i++) {
        if (UNION_FIND[i].parent == -1) {
            resultShifts[groupIdx] = UNION_FIND[i].shift;
            resultCounts[groupIdx] = UNION_FIND[i].count;
            groupIdx++;
        }
    }

    return groupIdx;
}
