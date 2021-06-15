#include "range_shifting.h"


static struct ShiftGroup UNION_FIND[150];


int8_t get_range_shift(FixedPoint value, uint8_t currentPrecision, uint8_t newWidth, uint8_t numShiftBits) {
    const int8_t nonFractional = 16 - currentPrecision;
    const uint8_t shiftOffset = (1 << (numShiftBits - 1));
    const uint16_t widthMask = (1 << newWidth) - 1;
    const uint16_t maxFp = (1 << (newWidth - 1)) - 1;
    const uint8_t newPrecision = maxZero8u(newWidth - nonFractional);

    volatile int8_t shift;
    volatile int8_t shiftedPrecision;
    volatile int8_t conversionShift;

    volatile FixedPoint shiftedValue;
    volatile FixedPoint error;

    volatile FixedPoint bestError = INT16_MAX;
    volatile int8_t bestShift = 0;

    uint16_t absValue = fp_abs(value);
    
    uint16_t i = 1 << numShiftBits;
    for (; i > 0; i--) {
        shift = i - shiftOffset;
        shiftedPrecision = newPrecision - shift;
        conversionShift = currentPrecision - shiftedPrecision;

        if (conversionShift > 0) {
            shiftedValue = absValue >> conversionShift;
        } else {
            conversionShift *= -1;
            shiftedValue = absValue << conversionShift;
        }

        error = fp_sub(maxFp, shiftedValue);

        if (error >= 0) {
            error = fp_abs(error);

            if (error < bestError) {
                bestError = error;
                bestShift = shift;
            }
        }
    }

    return bestShift;
}


void get_range_shifts_array(int8_t *result, FixedPoint *values, uint8_t currentPrecision, uint8_t newWidth, uint8_t numShiftBits, uint16_t numValues) {
    uint16_t i;
    for (i = 0; i < numValues; i++) {
        result[i] = get_range_shift(values[i], currentPrecision, newWidth, numShiftBits);
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


void merge(struct ShiftGroup *g1, struct ShiftGroup *g2, struct ShiftGroup *unionFind, uint16_t length) {
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

    // Update the score
    if (left->nextParent < length) {
        struct ShiftGroup *nextParent = unionFind + left->nextParent;
        int16_t count = left->count + nextParent->count;
        int16_t shiftDiff = fp_abs(left->shift - nextParent->shift);
        left->score = count + shiftDiff;
    } else {
        left->score = INT16_MAX;
    }
}


void get_groups_to_merge(uint16_t *result1, uint16_t *result2, struct ShiftGroup *unionFind, uint16_t numGroups) {
    // Locate the first parent
    uint16_t i = 0;
    while (unionFind[i].parent != -1) {
        i++;
    }

    // Get the initial left and right groups
    volatile struct ShiftGroup *left = unionFind + i;
    volatile struct ShiftGroup *right = unionFind + left->nextParent;
    
    if ((left->shift == right->shift) || (numGroups == 2)) {
        *result1 = left->idx;
        *result2 = right->idx;
        return;
    }

    volatile uint16_t bestLeft = i;
    volatile uint16_t bestRight = right->idx;
    volatile int16_t bestScore = left->score;
    volatile int16_t score;
    volatile uint16_t nextParent;

    left = right;
    nextParent = left->nextParent;
    right = unionFind + nextParent;

    while (nextParent < numGroups) {
        if (left->shift == right->shift) {
            *result1 = left->idx;
            *result2 = right->idx;
            return;
        }

        score = left->score;

        if (score < bestScore) {
            bestLeft = left->idx;
            bestRight = right->idx;
            bestScore = left->score;
        }

        left = right;
        nextParent = left->nextParent;
        right = unionFind + nextParent;
    }
    
    *result1 = bestLeft;
    *result2 = bestRight;
}


uint16_t create_shift_groups(int8_t *resultShifts, uint16_t *resultCounts, int8_t *shifts, uint16_t numShifts, uint16_t maxNumGroups) {
    // Create the initial groups via RLE
    uint16_t groupCount = run_length_encode_shifts(resultShifts, resultCounts, shifts, numShifts);

    if (groupCount <= maxNumGroups) {
        return groupCount;
    }

    // Initialize the union-find structure
    struct ShiftGroup *unionFind = UNION_FIND;
    int16_t countSum, shiftDiff;

    uint16_t i, j;
    for (i = 0; i < groupCount; i++) {
        j = i + 1;
        unionFind[i].idx = i;
        unionFind[i].count = resultCounts[i];
        unionFind[i].shift = resultShifts[i];
        unionFind[i].parent = -1;
        unionFind[i].nextParent = j;

        if (j < groupCount) {
            countSum = resultCounts[j] + resultCounts[i];
            shiftDiff = fp_abs(resultShifts[j] - resultShifts[i]);
            unionFind[i].score = countSum + shiftDiff;
        } else {
            unionFind[i].score = INT16_MAX;
        }
    }

    // Merge the groups until we meet the right amount
    uint16_t leftIdx, rightIdx;
    uint16_t numGroups = groupCount;

    while (numGroups > maxNumGroups) {
        get_groups_to_merge(&leftIdx, &rightIdx, unionFind, groupCount);
        merge(unionFind + leftIdx, unionFind + rightIdx, unionFind, groupCount);
        numGroups--;
    }

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
