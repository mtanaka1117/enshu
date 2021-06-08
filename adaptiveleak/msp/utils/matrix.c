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


struct Vector *vector_gated_add(struct Vector *result, struct Vector *vec1, struct Vector *vec2, FixedPoint gate, uint16_t precision) {
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
    if ((result->size != mat->numRows) || (vec->size != mat->numCols)) {
        return result;
    }

    uint16_t numRows = mat->numRows;
    uint16_t numCols = mat->numCols;

    uint16_t i, j;
    uint16_t row, col;
    FixedPoint prod, sum;

    for (i = numRows; i > 0; i--) {
        row = i - 1;
        sum = 0;

        for (j = numCols; j > 0; j--) {
            col = j - 1;

            prod = fp_mul(mat->data[MATRIX_INDEX(row, col, numCols)], vec->data[col], precision);
            sum = fp_add(sum, prod);
        }

        result->data[row] = sum;
    }

    return result;
}
