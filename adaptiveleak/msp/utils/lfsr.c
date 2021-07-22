#include "lfsr.h"


uint16_t lfsr(uint16_t start) {
    uint16_t val = start;
    uint16_t bit = ((val) ^ (val >> 2) ^ (val >> 3) ^ (val >> 5)) & 1;
    val = (val >> 1) | (bit << 15);
    return val;
}


void lfsr_array(uint8_t *startValues, uint8_t numValues) {
    uint16_t i;
    uint16_t tempValue;

    for (i = 0; i < numValues; i += 2) {
        tempValue = (uint16_t) startValues[i+1];
        tempValue = tempValue | (((uint16_t) startValues[i]) << 8);
        tempValue = lfsr(tempValue);

        startValues[i] = (tempValue >> 8) & 0xFF;
        startValues[i+1] = tempValue & 0xFF;
    }
}
