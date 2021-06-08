#include <stdint.h>
#include <stdio.h>
#include <assert.h>

#include "../utils/matrix.h"
#include "../utils/fixed_point.h"


#ifndef MATRIX_TESTS_H_
#define MATRIX_TESTS_H_

void test_add_four(void);
void test_add_ten(void);

void test_mul_four(void);
void test_mul_ten(void);

void test_gated_add_four(void);
void test_gated_add_ten(void);

void test_diff_norm_four(void);
void test_diff_norm_ten(void);
void test_diff_norm_overflow(void);

void test_norm_four(void);
void test_norm_ten(void);
void test_norm_overflow(void);

void test_absolute_diff_four(void);
void test_absolute_diff_ten(void);

void test_set_four(void);

void test_mat_vec_prod_3_4(void);
void test_mat_vec_prod_6_5(void);

uint8_t vector_equal(struct Vector *expected, struct Vector *given);

#endif
