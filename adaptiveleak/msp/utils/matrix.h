#include <stdint.h>
#include "fixed_point.h"

#ifndef MATRIX_H_
#define MATRIX_H_

struct Vector {
    FixedPoint *data;
    uint16_t size;
};


struct Vector *vector_add(struct Vector *result, struct Vector *vec1, struct Vector *vec2);
struct Vector *vector_mul(struct Vector *result, struct Vector *vec1, struct Vector *vec2, uint16_t precision);
struct Vector *vector_gated_add(struct Vector *result, struct Vector *vec1, struct Vector *vec2, FixedPoint gate, uint16_t precision);
FixedPoint vector_diff_norm(struct Vector *vec1, struct Vector *vec2);
void vector_set(struct Vector *vec, FixedPoint value);
struct Vector *vector_absolute_diff(struct Vector *result, struct Vector *vec1, struct Vector *vec2);
FixedPoint vector_norm(struct Vector *vec);

#endif
