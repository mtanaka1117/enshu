#include "math.h"


uint16_t min16u(uint16_t x, uint16_t y) {
    uint8_t comp = x <= y;
    return (comp * x) + ((1 - comp) * y);
}


uint8_t maxZero8u(uint8_t x) {
    return (uint8_t) ((x > 0) * x);
}


int8_t max8(int8_t x, int8_t y) {
    int8_t comp = x > y;
    return (comp * x) + (1 - comp) * y;
}


int8_t abs8(int8_t x) {
    return x * ((x > 0) - (x < 0));
}
