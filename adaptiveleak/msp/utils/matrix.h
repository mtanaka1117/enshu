#include <stdint.h>
#include "fixed_point.h"

#ifndef MATRIX_H_
#define MATRIX_H_

#define MATRIX_INDEX(X,Y,C) ((X) * (C) + (Y))

struct Vector {
    FixedPoint *data;
    uint16_t size;
};

struct Matrix {
    FixedPoint *data;
    uint16_t numRows;
    uint16_t numCols;
};


struct Vector *vector_add(struct Vector *result, struct Vector *vec1, struct Vector *vec2);
struct Vector *vector_mul(struct Vector *result, struct Vector *vec1, struct Vector *vec2, uint16_t precision);
struct Vector *vector_gated_add(struct Vector *result, struct Vector *vec1, struct Vector *vec2, FixedPoint gate, uint16_t precision);
FixedPoint vector_diff_norm(struct Vector *vec1, struct Vector *vec2);
void vector_set(struct Vector *vec, FixedPoint value);
struct Vector *vector_absolute_diff(struct Vector *result, struct Vector *vec1, struct Vector *vec2);
FixedPoint vector_norm(struct Vector *vec);
struct Vector *matrix_vector_prod(struct Vector *result, struct Matrix *mat, struct Vector *vec, uint16_t precision);

#endif
