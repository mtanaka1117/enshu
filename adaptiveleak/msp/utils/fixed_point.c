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


FixedPoint fp_neg(FixedPoint x) {
    return -1 * x;
}


FixedPoint fp_abs(FixedPoint x) {
    return x * ((x > 0) - (x < 0));
}


FixedPoint fp32_mul(FixedPoint x, FixedPoint y, uint16_t precision) {
    int32_t xLarge = (int32_t) x;
    int32_t yLarge = (int32_t) y;
    int32_t result = (x * y) >> precision;
    return (FixedPoint) result;
}

int16_t fp_tanh(int16_t x, uint16_t precision) {
    /**
     * Approximates tanh using a polynomial.
     */
    uint8_t shouldInvertSign = 0;
    if (x < 0) {
        x = fp_neg(x);
        shouldInvertSign = 1;
    }
    
    FixedPoint fourth = 1 << (precision - 2);
    FixedPoint half = 1 << (precision - 1);
    FixedPoint one = 1 << precision;
    FixedPoint two = 1 << (precision + 1);

    // Approximate tanh(x) using a piece-wise linear function
    FixedPoint result = one;
    if (x <= fourth) {
        result = x;
    } else if (x <= 3 * fourth) {
        result = 3 * (x >> 2) + 5 * (1 << (precision - 6));
    } else if (x <= (one + fourth)) {
        result = (x >> 1) + fourth;
    } else if (x <= (two + fourth)) {
        result = (x >> 3) + (half + fourth - (1 << (precision - 5)));

        if (result > one) {
            result = one;
        }
    }
    
    if (shouldInvertSign) {
        return fp_neg(result);
    }
    return result;
}


int16_t fp_sigmoid(int16_t x, uint16_t precision) {
    /**
     * Approximates the sigmoid function using tanh.
     */
    uint8_t should_invert_sign = 0;
    if (x < 0) {
        x = fp_neg(x);
        should_invert_sign = 1;
    }

    FixedPoint one = 1 << precision;
    FixedPoint tanh = fp_tanh(x >> 1, precision);
    FixedPoint result = fp_add(tanh, one) >> 1;

    if (should_invert_sign) {
        result = one - result;
    }

    return result;
}
