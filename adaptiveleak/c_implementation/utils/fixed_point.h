#include <stdint.h>

#ifndef FIXED_POINT_H_
#define FIXED_POINT_H_

typedef int16_t FixedPoint;

FixedPoint fp_add(FixedPoint x, FixedPoint y);
FixedPoint fp_mul(FixedPoint x, FixedPoint y, uint16_t precision);
FixedPoint fp_abs(FixedPoint x);
FixedPoint fp_sub(FixedPoint x, FixedPoint y);
FixedPoint fp_neg(FixedPoint x);
FixedPoint fp32_mul(FixedPoint x, FixedPoint y, uint16_t precision);
FixedPoint fp_sigmoid(FixedPoint x, uint16_t precision);
FixedPoint fp_tanh(FixedPoint x, uint16_t precision);
FixedPoint fp_convert(FixedPoint x, uint16_t oldPrecision, uint16_t newWidth, uint16_t newPrecision);

void fp_convert_array(FixedPoint *array, uint16_t oldPrecision, uint16_t newPrecision, uint16_t newWidth, uint16_t startIdx, uint16_t length);

#endif
