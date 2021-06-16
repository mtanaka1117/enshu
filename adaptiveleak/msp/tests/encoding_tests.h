#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "../utils/encoding.h"
#include "../utils/encryption.h"
#include "../utils/bitmap.h"
#include "../utils/matrix.h"
#include "../utils/pruning.h"

#ifndef ENCODING_TESTS_H_
#define ENCODING_TESTS_H_

// Encoding collected indices
void test_collected_indices_three(void);
void test_collected_indices_ten(void);
void test_collected_indices_23(void);
void test_collected_indices_ten_unset(void);

// Standard Encoding Tests
void test_standard_encode_four(void);
void test_standard_encode_ten(void);

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

// Testing Measurement Pruning
void test_pruning_4_1(void);
void test_pruning_4_2(void);

// Test Group Encoding
void test_group_encode_four(void);

#endif
