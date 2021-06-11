#include "matrix.h"


#ifdef IS_MSP
#include "DSPLib.h"

// For MSP implementations, we allocate memory in the LEA RAM.
// This memory is used when executing Matrix multiplications.
DSPLIB_DATA(MULTIPLY_BUFFER, 4);
static FixedPoint MULTIPLY_BUFFER[1800];

FixedPoint *dma_load(FixedPoint *result, FixedPoint *data, uint16_t n) {
    /**
     * Loads the first n elements of the data array into the result array using
     * DMA.
     */
    // Configure DMA channel 0
    __data20_write_long((uintptr_t) &DMA0SA, (uintptr_t) data);   // Source block address
    __data20_write_long((uintptr_t) &DMA0DA, (uintptr_t) result); // Destination single address
    DMA0SZ = n;                                      // Block size
    DMA0CTL = DMADT_5 | DMASRCINCR_3 | DMADSTINCR_3; // Rpt, inc
    DMA0CTL |= DMAEN;                                // Enable DMA0
    DMA0CTL |= DMAREQ;

    return result;
}
#endif


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

    volatile FixedPoint temp1;
    volatile FixedPoint temp2;

    const FixedPoint one = 1 << precision;
    volatile FixedPoint oneMinusGate;
    volatile FixedPoint gateValue;

    uint16_t i;
    for (i = 0; i < vec1->size; i++) {
        gateValue = gate->data[i];
        oneMinusGate = fp_sub(one, gateValue);

        temp1 = fp_mul(vec1->data[i], gateValue, precision);
        temp2 = fp_mul(vec2->data[i], oneMinusGate, precision);
        result->data[i] = fp_add(temp1, temp2);
    }

    return result;
}

//struct Vector *vector_gated_add(struct Vector *result, struct Vector *vec1, struct Vector *vec2, struct Vector *gate, uint16_t precision) {
//    /**
//     * Returns a vector with gate * vec1 + (1 - gate) * vec2
//     */
//    if ((vec1->size != vec2->size) || (vec1->size != result->size) || (vec1->size != gate->size)) {
//        return result;
//    }
//
//    uint16_t i, j;
//    FixedPoint temp1, temp2;
//
//    FixedPoint one = 1 << precision;
//    FixedPoint oneMinusGate, gateValue;
//
//    for (i = vec1->size; i > 0; i--) {
//        j = i - 1;
//        
//        gateValue = gate->data[j];
//        oneMinusGate = fp_sub(one, gateValue);
//
//        temp1 = fp_mul(vec1->data[j], gateValue, precision);
//        temp2 = fp_mul(vec2->data[j], oneMinusGate, precision);
//        result->data[j] = fp_add(temp1, temp2);
//    }
//
//    return result;
//}


void vector_set(struct Vector *vec, FixedPoint value) {
    uint16_t i;
    for (i = vec->size; i > 0; i--) {
        vec->data[i - 1] = value;
    }
}


void vector_copy(struct Vector *dst, struct Vector *src) {
    if (dst->size != src->size) {
        return;
    }

    uint16_t i, j;
    for (i = dst->size; i > 0; i--) {
        j = i - 1;
        dst->data[j] = src->data[j];
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

    #ifdef IS_MSP
    // First transfer the input matrices to the LEA RAM segment using DMA
    uint16_t offset = 0;
    FixedPoint *vecData = dma_load(MULTIPLY_BUFFER, vec->data, numRows);
    offset += numRows * 2;  // Ensure we have room for 2 columns, as the LEA requires "even" dimensions

    FixedPoint *matData = dma_load(MULTIPLY_BUFFER + offset, mat->data, numRows * numCols);
    offset += numRows * numCols;

    FixedPoint *resultData = MULTIPLY_BUFFER + offset;  // Temporary buffer (in LEA RAM) for the result

    // When using the MSP430, we use the LEA for Matrix multiplications. Based on profiling,
    // the LEA can take up to 5x fewer compute cycles than a standard implementation.
    msp_status status;
    msp_matrix_mpy_q15_params mulParams;

    // Initialze LEA metadata
    mulParams.srcARows = 2;
    mulParams.srcACols = numRows;
    mulParams.srcBRows = numRows;
    mulParams.srcBCols = numCols;

    // Perform Matrix multiplication using the LEA
    status = msp_matrix_mpy_q15(&mulParams, vecData, matData, resultData);
    msp_checkStatus(status);

    // Convert back to the original fixed-point precision. The LEA assumes 15 fractional bits.
    msp_matrix_shift_q15_params shiftParams;
    shiftParams.rows = 2;
    shiftParams.cols = numCols;
    shiftParams.shift = 15 - precision;

    // Perform element-wise shift using the LEA
    if (shiftParams.shift > 0) {
        status = msp_matrix_shift_q15(&shiftParams, resultData, resultData);
        msp_checkStatus(status);
    }

    // Load result back into the given result Matrix. We omit
    // any padding elements in this transfer.
    dma_load(result->data, resultData, numCols);

    #else
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
    #endif

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


struct Vector *vector_scale(struct Vector *result, struct Vector *vec, struct Vector *mean, struct Vector *scale, uint16_t inPrecision, uint16_t outPrecision) {
    if ((result->size != vec->size) || (result->size != mean->size) || (result->size != scale->size)) {
        return result;
    }

    uint8_t shouldShiftRight = (inPrecision > outPrecision);
    uint16_t shiftAmount = inPrecision - outPrecision;
    shiftAmount = shouldShiftRight * shiftAmount + (1 - shouldShiftRight) * (-1 * shiftAmount);

    uint16_t i, j;
    FixedPoint diff, normalized;

    for (i = result->size; i > 0; i--) {
        j = i - 1;
        diff = fp_sub(vec->data[j], mean->data[j]);
        normalized = fp32_mul(diff, scale->data[j], inPrecision);

        if (shouldShiftRight) {
            result->data[j] = normalized >> shiftAmount;
        } else {
            result->data[j] = normalized << shiftAmount;
        }
    }

    return result;
}
