#include <stdint.h>
#include <assert.h>
#include <stdio.h>

#include "../utils/fixed_point.h"
#include "../utils/range_shifting.h"

#ifndef RANGE_SHIFTING_TESTS_H_
#define RANGE_SHIFTING_TESTS_H_

void test_range_125_13(void);
void test_range_225_13(void);
void test_range_shift_large(void);
void test_range_shift_large_2(void);
void test_range_neg125_13(void);
void test_range_1125_10(void);
void test_range_9125_10(void);
void test_range_shifts_array(void);

void test_convert_1_10(void);
void test_convert_125_10(void);
void test_convert_425_12(void);
void test_convert_neg_1_10(void);

void test_rle_4(void);
void test_rle_5(void);
void test_rle_6(void);

void test_union_find_simple(void);

void test_grouping_first_4(void);
void test_grouping_twice_5(void);
void test_grouping_twice_6(void);
void test_grouping_three_6(void);

#endif
