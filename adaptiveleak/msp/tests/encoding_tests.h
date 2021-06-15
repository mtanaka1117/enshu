#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "../utils/encoding.h"
#include "../utils/encryption.h"
#include "../utils/bitmap.h"
#include "../utils/matrix.h"

#ifndef ENCODING_TESTS_H_
#define ENCODING_TESTS_H_

// Encoding collected indices
void test_collected_indices_three();
void test_collected_indices_ten();
void test_collected_indices_23();
void test_collected_indices_ten_unset();

// Standard Encoding Tests
void test_standard_encode_four();
void test_standard_encode_ten();

// Rounding to block sizes
void test_rounding_equal(void);
void test_rounding_five(void);
void test_rounding_ten(void);

// Test Group Message Length Calculations
void test_group_length_block_1(void);
void test_group_length_block_2(void);

// Test Group Bit Width Setting
void test_set_widths_3(void);
void test_set_widths_4(void);
void test_set_widths_5(void);

#endif
