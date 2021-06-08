#include <stdint.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "../utils/encoding.h"
#include "../utils/bitmap.h"

#ifndef ENCODING_TESTS_H_
#define ENCODING_TESTS_H_

// Encoding collected indices
void test_collected_indices_three();
void test_collected_indices_ten();
void test_collected_indices_23();

// Standard Encoding Tests
void test_standard_encode_three();

#endif
