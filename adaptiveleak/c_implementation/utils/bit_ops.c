#include "bit_ops.h"


uint8_t get_most_significant_place(int16_t value) {
    // Get the sign and take the absolute value
    int16_t sign = ((value >= 0) << 1) - 1;
    uint16_t absValue = (uint16_t) (sign * value);

    uint8_t result = 0;
    uint8_t currentPlace = 0;

    while (absValue > 0) {
        if ((absValue & 1) == 1) {
            result = currentPlace;
        }

        absValue = absValue >> 1;
        currentPlace++;
    }

    return result;
}


int16_t extract_bit_range(int16_t value, uint8_t startPos, uint8_t length) {
    // Get the sign an absolute value
    int16_t sign = ((value >= 0) << 1) - 1;
    uint16_t absValue = (uint16_t) (sign * value);

    uint16_t mask = (1 << length) - 1;
    mask = ~(mask << startPos);

    int16_t absResult = (int16_t) (absValue & mask);
    return sign * absResult;
}
