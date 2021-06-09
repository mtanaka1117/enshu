#include "matrix.h"


struct Vector *vector_add(struct Vector *result, struct Vector *vec1, struct Vector *vec2) {
    /**
     * Adds vec1 and vec2 in an element-wise manner.
     */
    if ((vec1->size != vec2->size) || (vec1->size != result->size)) {
        return result;
    }

    uint16_t i, j;
    for (i = vec1->size; i > 0; i--) {
        j = i - 1;
        result->data[j] = fp_add(vec1->data[j], vec2->data[j]);
    }

    return result;
}


struct Vector *vector_mul(struct Vector *result, struct Vector *vec1, struct Vector *vec2, uint16_t precision) {
    /**
     * Multiplies the given vectors in an element-wise manner.
     */
    if ((vec1->size != vec2->size) || (vec1->size != result->size)) {
        return result;
    }

    uint16_t i, j;
    for (i = vec1->size; i > 0; i--) {
        j = i - 1;
        result->data[j] = fp_mul(vec1->data[j], vec2->data[j], precision);
    }

    return result;
}


struct Vector *vector_gated_add_scalar(struct Vector *result, struct Vector *vec1, struct Vector *vec2, FixedPoint gate, uint16_t precision) {
    /**
     * Returns a vector with gate * vec1 + (1 - gate) * vec2
     */
    if ((vec1->size != vec2->size) || (vec1->size != result->size)) {
        return result;
    }

    uint16_t i, j;
    FixedPoint temp1, temp2;

    FixedPoint oneMinusGate = fp_sub(1 << precision, gate);

    for (i = vec1->size; i > 0; i--) {
        j = i - 1;
        
        temp1 = fp_mul(vec1->data[j], gate, precision);
        temp2 = fp_mul(vec2->data[j], oneMinusGate, precision);
        result->data[j] = fp_add(temp1, temp2);
    }

    return result;
}


struct Vector *vector_gated_add(struct Vector *result, struct Vector *vec1, struct Vector *vec2, struct Vector *gate, uint16_t precision) {
    /**
     * Returns a vector with gate * vec1 + (1 - gate) * vec2
     */
    if ((vec1->size != vec2->size) || (vec1->size != result->size) || (vec1->size != gate->size)) {
        return result;
    }

    uint16_t i, j;
    FixedPoint temp1, temp2;

    FixedPoint one = 1 << precision;
    FixedPoint oneMinusGate, gateValue;

    for (i = vec1->size; i > 0; i--) {
        j = i - 1;
        
        gateValue = gate->data[j];
        oneMinusGate = fp_sub(one, gateValue);

        temp1 = fp_mul(vec1->data[j], gateValue, precision);
        temp2 = fp_mul(vec2->data[j], oneMinusGate, precision);
        result->data[j] = fp_add(temp1, temp2);
    }

    return result;
}


void vector_set(struct Vector *vec, FixedPoint value) {
    uint16_t i;
    for (i = vec->size; i > 0; i--) {
        vec->data[i - 1] = value;
    }
}


struct Vector *vector_absolute_diff(struct Vector *result, struct Vector *vec1, struct Vector *vec2) {
    if ((vec1->size != vec2->size) || (result->size != vec1->size)) {
        return result;
    }

    uint16_t i, j;
    for (i = vec1->size; i > 0; i--) {
        j = i - 1;
        result->data[j] = fp_abs(fp_sub(vec1->data[j], vec2->data[j]));
    }

    return result;
}


FixedPoint vector_norm(struct Vector *vec) {
    FixedPoint value = 0;
    FixedPoint norm = 0;

    uint16_t i;
    for (i = vec->size; i > 0; i--) {
        value = fp_abs(vec->data[i - 1]);

        // Protect against overflow
        if ((INT16_MAX - norm) < value) {
            return INT16_MAX;
        }

        // Add value into the running norm
        norm = fp_add(norm, value);
    }

    return norm;
}


FixedPoint vector_diff_norm(struct Vector *vec1, struct Vector *vec2) {
    if (vec1->size != vec2->size) {
        return 0;
    }

    FixedPoint diff = 0;
    FixedPoint norm = 0;
    
    uint16_t i, j;
    for (i = vec1->size; i > 0; i--) {
        j = i - 1;

        diff = fp_abs(fp_sub(vec1->data[j], vec2->data[j]));

        // Protect against overflow
        if ((INT16_MAX - norm) < diff) {
            return INT16_MAX;
        }

        // Add value into the running norm
        norm = fp_add(norm, diff);
    }

    return norm;
}


struct Vector *matrix_vector_prod(struct Vector *result, struct Matrix *mat, struct Vector *vec, uint16_t precision) {
    /**
     * Computes the product v^T * M
     */
    if ((result->size != mat->numCols) || (vec->size != mat->numRows)) {
        return result;
    }

    uint16_t numRows = mat->numRows;
    uint16_t numCols = mat->numCols;

    uint16_t i, j;
    uint16_t row, col;
    FixedPoint prod, sum;

    for (i = numCols; i > 0; i--) {
        col = i - 1;
        sum = 0;

        for (j = numRows; j > 0; j--) {
            row = j - 1;

            prod = fp_mul(mat->data[MATRIX_INDEX(row, col, numCols)], vec->data[row], precision);
            sum = fp_add(sum, prod);
        }

        result->data[col] = sum;
    }

    return result;
}


FixedPoint vector_dot_prod(struct Vector *vec1, struct Vector *vec2, uint16_t precision) {
    if (vec1->size != vec2->size) {
        return 0;
    }

    FixedPoint result = 0;

    uint16_t i, j;
    for (i = vec1->size; i > 0; i--) {
        j = i - 1;
        result = fp_add(result, fp_mul(vec1->data[j], vec2->data[j], precision));
    }

    return result;
}


struct Vector *vector_apply(struct Vector *result, struct Vector *vec, FixedPoint (*fn)(FixedPoint, uint16_t), uint16_t precision) {
    if (vec->size != result->size) {
        return result;
    }

    uint16_t i, j;
    for (i = vec->size; i > 0; i--) {
        j = i - 1;
        result->data[j] = fn(vec->data[j], precision);
    }

    return result;
}


struct Vector *vector_stack(struct Vector *result, struct Vector *first, struct Vector *second) {
    if (result->size != first->size + second->size) {
        return result;
    }

    uint16_t i, j;
    for (i = first->size; i > 0; i--) {
        j = i - 1;
        result->data[j] = first->data[j];
    }

    uint16_t offset = first->size;
    for (i = second->size; i > 0; i--) {
        j = i - 1;
        result->data[j + offset] = second->data[j];
    }

    return result;
}


struct Vector *vector_scale(struct Vector *result, struct Vector *vec, struct Vector *mean, struct Vector *scale, uint16_t precision) {
    if ((result->size != vec->size) || (result->size != mean->size) || (result->size != scale->size)) {
        return result;
    }

    uint16_t i, j;
    FixedPoint diff;

    for (i = result->size; i > 0; i--) {
        j = i - 1;
        diff = fp_sub(vec->data[j], mean->data[j]);
        result->data[j] = fp32_mul(diff, scale->data[j], precision);
    }

    return result;
}
