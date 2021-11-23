#include "range_shifting_tests.h"


int main(void) {
    test_range_1_3();
    printf("Passed all tests.\n");

    return 0;
}


void test_range_1_3(void) {
    uint8_t currentPrecision = 13;
    uint8_t newWidth = 4;
    uint8_t numShiftBits = 3;
    FixedPoint value = (1 << currentPrecision) + (1 << (currentPrecision - 2));  // 1.25

    int8_t shift = get_range_shift(value, currentPrecision, newWidth, numShiftBits);
    assert(shift == 1);
}

