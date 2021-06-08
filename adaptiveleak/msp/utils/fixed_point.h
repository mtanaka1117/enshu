#include <stdint.h>

#ifndef FIXED_POINT_H_
#define FIXED_POINT_H_

typedef int16_t FixedPoint;

FixedPoint fp_add(FixedPoint x, FixedPoint y);
FixedPoint fp_mul(FixedPoint x, FixedPoint y, uint16_t precision);
FixedPoint fp_abs(FixedPoint x);
FixedPoint fp_sub(FixedPoint x, FixedPoint y);
FixedPoint fp_norm(FixedPoint *array, uint16_t length);

#endif
