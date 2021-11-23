#include <stdint.h>

#ifndef BIT_OPS_H_
#define BIT_OPS_H

#define SET_BIT(X, Y)      ((X) |= (Y))
#define CLEAR_BIT(X, Y)    ((X) &= ~(Y))
#define TOGGLE_BIT(X, Y)   ((X) ^= (Y))
#define TEST_BIT(X, Y)     ((X) & (Y))


uint8_t get_most_significant_place(int16_t value);
int16_t extract_bit_range(int16_t value, uint8_t startPos, uint8_t length);


#endif
