#include <stdint.h>
#include "fixed_point.h"

#ifndef MATRIX_H_
#define MATRIX_H_

#define MATRIX_INDEX(X,Y,C) ((X) * (C) + (Y))
#define USE_LEA

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

struct Vector *vector_gated_add(struct Vector *result, struct Vector *vec1, struct Vector *vec2, struct Vector *gate, uint16_t precision);
struct Vector *vector_gated_add_scalar(struct Vector *result, struct Vector *vec1, struct Vector *vec2, FixedPoint gate, uint16_t precision);

FixedPoint vector_diff_norm(struct Vector *vec1, struct Vector *vec2);
void vector_set(struct Vector *vec, FixedPoint value);
struct Vector *vector_absolute_diff(struct Vector *result, struct Vector *vec1, struct Vector *vec2);
FixedPoint vector_norm(struct Vector *vec);

struct Vector *matrix_vector_prod(struct Vector *result, struct Matrix *mat, struct Vector *vec, uint16_t precision);
FixedPoint vector_dot_prod(struct Vector *vec1, struct Vector *vec2, uint16_t precision);

struct Vector *vector_stack(struct Vector *result, struct Vector *first, struct Vector *second);
struct Vector *vector_scale(struct Vector *result, struct Vector *vec, struct Vector *mean, struct Vector *scale, uint16_t inPrecision, uint16_t outPrecision);
void vector_copy(struct Vector *dst, struct Vector *src);
struct Vector *vector_apply(struct Vector *result, struct Vector *vec, FixedPoint (*fn)(FixedPoint, uint16_t), uint16_t precision);

#endif
