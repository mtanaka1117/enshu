#include "range_shifting_tests.h"


int main(void) {
    printf("===== Testing Range Shifts =====\n");
    test_range_125_13();
    test_range_225_13();
    test_range_neg125_13();
    test_range_1125_10();
    test_range_9125_10();
    test_range_shift_large();
    test_range_shifts_array();
    printf("\tPassed Shifting Tests.\n");

    printf("===== Testing Fixed Point Conversion =====\n");
    test_convert_1_10();
    test_convert_125_10();
    test_convert_425_12();
    test_convert_neg_1_10();
    printf("\tPassed Conversion Tests.\n");

    printf("==== Testing Run Length Encoding ====\n");
    test_rle_4();
    test_rle_5();
    test_rle_6();
    printf("\tPassed Run Length Encoding Tests.\n");

    printf("==== Testing Union Find ====\n");
    test_union_find_simple();
    printf("\tPassed Union Find Tests.\n");

    printf("==== Testing Shift Grouping ====\n");
    test_grouping_first_4();
    test_grouping_twice_5();
    test_grouping_twice_6();
    test_grouping_three_6();
    printf("\tPassed Grouping Tests.\n");

    return 0;
}


void test_range_125_13(void) {
    uint8_t currentPrecision = 13;
    uint8_t newWidth = 4;
    uint8_t numShiftBits = 3;
    FixedPoint value = (1 << currentPrecision) + (1 << (currentPrecision - 2));  // 1.25

    int8_t shift = get_range_shift(value, currentPrecision, newWidth, numShiftBits);
    assert(shift == -1);
}


void test_range_225_13(void) {
    uint8_t currentPrecision = 13;
    uint8_t newWidth = 4;
    uint8_t numShiftBits = 3;
    FixedPoint value = (1 << (currentPrecision + 1)) + (1 << (currentPrecision - 2));  // 2.25

    int8_t shift = get_range_shift(value, currentPrecision, newWidth, numShiftBits);
    assert(shift == 0);
}


void test_range_neg125_13(void) {
    uint8_t currentPrecision = 13;
    uint8_t newWidth = 4;
    uint8_t numShiftBits = 3;
    FixedPoint value = fp_neg((1 << currentPrecision) + (1 << (currentPrecision - 2)));  // 1.25

    int8_t shift = get_range_shift(value, currentPrecision, newWidth, numShiftBits);
    assert(shift == -1);
}


void test_range_1125_10(void) {
    uint8_t currentPrecision = 10;
    uint8_t newWidth = 6;
    uint8_t numShiftBits = 3;
    FixedPoint value = (1 << currentPrecision) + (1 << (currentPrecision - 3));  // 1.125

    int8_t shift = get_range_shift(value, currentPrecision, newWidth, numShiftBits);
    assert(shift == -3);
}


void test_range_9125_10(void) {
    uint8_t currentPrecision = 10;
    uint8_t newWidth = 6;
    uint8_t numShiftBits = 3;
    FixedPoint value = (1 << (currentPrecision + 3)) + (1 << currentPrecision) + (1 << (currentPrecision - 3));  // 9.125

    int8_t shift = get_range_shift(value, currentPrecision, newWidth, numShiftBits);
    assert(shift == -1);
}


void test_range_shift_large(void) {
    uint8_t currentPrecision = 10;
    uint8_t newWidth = 15;
    uint8_t numShiftBits = 3;
    FixedPoint value = 0x5A47;

    int8_t shift = get_range_shift(value, currentPrecision, newWidth, numShiftBits);
    assert(shift == 0);
}


void test_range_shifts_array(void) {
    uint8_t currentPrecision = 13;
    uint8_t newWidth = 4;
    uint8_t numShiftBits = 3;

    FixedPoint values[4];
    values[0] = (1 << (currentPrecision + 1)) + (1 << (currentPrecision - 2));  // 2.25
    values[1] = (1 << currentPrecision) + (1 << (currentPrecision - 2));  // 1.25
    values[2] = fp_neg(values[1]);  // -1.25
    values[3] = (1 << currentPrecision) + (1 << (currentPrecision - 4));  // 1.0625

    int8_t shifts[4];
    get_range_shifts_array(shifts, values, currentPrecision, newWidth, numShiftBits, 4);

    assert(shifts[0] == 0);
    assert(shifts[1] == -1);
    assert(shifts[2] == -1);
    assert(shifts[3] == -1);
}


void test_convert_1_10(void) {
    uint16_t oldPrecision = 10;
    uint16_t newPrecision = 1;
    uint16_t newWidth = 4;
    FixedPoint one = 1 << oldPrecision;

    FixedPoint result = fp_convert(one, oldPrecision, newPrecision, newWidth);
    assert(result == 2);
}


void test_convert_125_10(void) {
    uint16_t oldPrecision = 10;
    uint16_t newPrecision = 2;
    uint16_t newWidth = 4;
    FixedPoint value = (1 << oldPrecision) + (1 << (oldPrecision - 2));  // 1.25

    FixedPoint result = fp_convert(value, oldPrecision, newPrecision, newWidth);
    assert(result == 5);
}


void test_convert_425_12(void) {
    uint16_t oldPrecision = 12;
    uint16_t newPrecision = 1;
    uint16_t newWidth = 5;
    FixedPoint value = (1 << (oldPrecision + 2)) + (1 << (oldPrecision - 2));  // 4.25

    FixedPoint result = fp_convert(value, oldPrecision, newPrecision, newWidth);
    assert(result == 8);
}


void test_convert_neg_1_10(void) {
    uint16_t oldPrecision = 10;
    uint16_t newPrecision = 1;
    uint16_t newWidth = 4;
    FixedPoint one = fp_neg(1 << oldPrecision);

    FixedPoint result = fp_convert(one, oldPrecision, newPrecision, newWidth);
    assert(result == 14);
}


void test_rle_4(void) {
    int8_t shifts[4] = { -1, -1, 1, 0 };
    int8_t resultShifts[4] = { -10, -10, -10, -10 };
    uint16_t resultCounts[4] = { 100, 100, 100, 100 };

    uint16_t count = run_length_encode_shifts(resultShifts, resultCounts, shifts, 4);
    assert(count == 3);

    assert(resultShifts[0] == -1);
    assert(resultShifts[1] == 1);
    assert(resultShifts[2] == 0);

    assert(resultCounts[0] == 2);
    assert(resultCounts[1] == 1);
    assert(resultCounts[2] == 1);
}


void test_rle_5(void) {
    int8_t shifts[5] = { -1, -1, 1, 1, 1 };
    int8_t resultShifts[5] = { -10, -10, -10, -10, -10 };
    uint16_t resultCounts[5] = { 100, 100, 100, 100, 100 };

    uint16_t count = run_length_encode_shifts(resultShifts, resultCounts, shifts, 5);
    assert(count == 2);

    assert(resultShifts[0] == -1);
    assert(resultShifts[1] == 1);

    assert(resultCounts[0] == 2);
    assert(resultCounts[1] == 3);
}


void test_rle_6(void) {
    int8_t shifts[6] = { 0, -1, -1, 1, 1, 1 };
    int8_t resultShifts[6] = { -10, -10, -10, -10, -10, -10 };
    uint16_t resultCounts[6] = { 100, 100, 100, 100, 100, 100 };

    uint16_t count = run_length_encode_shifts(resultShifts, resultCounts, shifts, 6);
    assert(count == 3);

    assert(resultShifts[0] == 0);
    assert(resultShifts[1] == -1);
    assert(resultShifts[2] == 1);

    assert(resultCounts[0] == 1);
    assert(resultCounts[1] == 2);
    assert(resultCounts[2] == 3);
}


void test_union_find_simple(void) {
    // Initialize the union find structure
    struct ShiftGroup unionFind[4];

    struct ShiftGroup g0 = { 0, -1, 1, 2, 4, 2 };
    unionFind[0] = g0;

    struct ShiftGroup g1 = { 1, 0, 1, 2, 3, 1 };
    unionFind[1] = g1;

    struct ShiftGroup g2 = { 2, -1, 0, 3, 3, 1 };
    unionFind[2] = g2;

    struct ShiftGroup g3 = { 3, -1, -1, 4, INT16_MAX, 1 };
    unionFind[3] = g3;

    assert(find(&g0, unionFind) == 0);
    assert(find(&g1, unionFind) == 0);
    assert(find(&g2, unionFind) == 2);
    assert(find(&g3, unionFind) == 3);

    merge(&unionFind[0], &unionFind[2], unionFind, 4);
    assert(unionFind[0].nextParent == 3);
    assert(unionFind[0].shift == 1);
    assert(unionFind[0].count == 3);
    assert(unionFind[0].score == 6);
    assert(unionFind[2].parent == 0);

    merge(&unionFind[2], &unionFind[3], unionFind, 4);
    assert(unionFind[0].nextParent == 4);
    assert(unionFind[0].score == INT16_MAX);
    assert(unionFind[0].count == 4);
    assert(unionFind[0].shift == 1);
    assert(unionFind[3].parent == 0);
}


void test_grouping_first_4(void) {
    int8_t shifts[4] = {0, -1, -2, -2};
    int8_t resultShifts[4];
    uint16_t resultCounts[4];

    uint16_t numGroups = create_shift_groups(resultShifts, resultCounts, shifts, 4, 2);

    assert(numGroups == 2);
    assert(resultShifts[0] == 0);
    assert(resultShifts[1] == -2);
    assert(resultCounts[0] == 2);
    assert(resultCounts[1] == 2);
}


void test_grouping_twice_5(void) {
    int8_t shifts[5] = {0, -1, -2, -3, -2};
    int8_t resultShifts[5];
    uint16_t resultCounts[5];

    uint16_t numGroups = create_shift_groups(resultShifts, resultCounts, shifts, 5, 2);

    assert(numGroups == 2);
    assert(resultShifts[0] == 0);
    assert(resultShifts[1] == -2);
    assert(resultCounts[0] == 2);
    assert(resultCounts[1] == 3);
}


void test_grouping_twice_6(void) {
    int8_t shifts[6] = {0, -1, 3, 2, 3, 2};
    int8_t resultShifts[6];
    uint16_t resultCounts[6];

    uint16_t numGroups = create_shift_groups(resultShifts, resultCounts, shifts, 6, 2);

    assert(numGroups == 2);
    assert(resultShifts[0] == 0);
    assert(resultShifts[1] == 3);
    assert(resultCounts[0] == 2);
    assert(resultCounts[1] == 4);
}


void test_grouping_three_6(void) {
    int8_t shifts[6] = {0, -1, 3, 2, 3, 2};
    int8_t resultShifts[6];
    uint16_t resultCounts[6];

    uint16_t numGroups = create_shift_groups(resultShifts, resultCounts, shifts, 6, 3);

    assert(numGroups == 3);
    assert(resultShifts[0] == 0);
    assert(resultShifts[1] == 3);
    assert(resultShifts[2] == 2);

    assert(resultCounts[0] == 2);
    assert(resultCounts[1] == 3);
    assert(resultCounts[2] == 1);
}
