#include "fixed_point.h"


FixedPoint fp_add(FixedPoint x, FixedPoint y) {
    return x + y;
}


FixedPoint fp_sub(FixedPoint x, FixedPoint y) {
    return x - y;
}


FixedPoint fp_mul(FixedPoint x, FixedPoint y, uint16_t precision) {
    return (x * y) >> precision;
}


FixedPoint fp_abs(FixedPoint x) {
    return x * ((x > 0) - (x < 0));
}


FixedPoint fp_norm(FixedPoint *array, uint16_t length) {
    uint16_t i, j;

    FixedPoint arrayValue;
    FixedPoint norm = 0;

    for (i = length; i > 0; i--) {
        arrayValue = array[i - 1];

        // Protect against overflow
        if ((INT16_MAX - norm) < arrayValue) {
            return INT16_MAX;
        }

        // Compute the absolute value of the array element
        arrayValue = fp_abs(arrayValue);

        // Add value into the running norm
        norm = fp_add(norm, arrayValue);
    }

    return norm;
}
