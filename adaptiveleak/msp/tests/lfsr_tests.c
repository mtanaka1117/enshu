#include "lfsr_tests.h"


int main(void) {

    printf("===== Testing Period =====\n");
    test_lfsr_period();
    printf("\tPassed Period tests.\n");

    return 0;
}


void test_lfsr_period(void) {
    uint16_t counter = 0;
    uint16_t startValue = 0xACE1u;
    uint16_t lfsrValue = startValue;

    do {
        lfsrValue = lfsr(lfsrValue, 1);
        counter++;
    } while (startValue != lfsrValue);

    assert(counter == 65535);
}

