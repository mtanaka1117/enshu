#include "matrix_tests.h"


int main(void) {
    printf("==== Testing Vector Set ====\n");
    test_set_four();
    printf("\tPassed Vector Set Tests.\n");

    printf("==== Testing Vector Add ====\n");
    test_add_four();
    test_add_ten();
    printf("\tPassed Vector Add Tests.\n");

    printf("==== Testing Vector Multiply ====\n");
    test_mul_four();
    test_mul_ten();
    printf("\tPassed Vector Multiply Tests.\n");

    printf("==== Testing Vector Gated Add ====\n");
    test_gated_add_four();
    test_gated_add_ten();
    printf("\tPassed Vector Gated Add Tests.\n");

    printf("==== Testing Vector Norm ====\n");
    test_norm_four();
    test_norm_ten();
    test_norm_overflow();
    printf("\tPassed Vector Norm Tests.\n");

    printf("==== Testing Vector Absolute Diff ====\n");
    test_absolute_diff_four();
    test_absolute_diff_ten();
    printf("\tPassed Vector Absolute Diff Tests.\n");

    printf("==== Testing Vector Diff Norm ====\n");
    test_diff_norm_four();
    test_diff_norm_ten();
    test_diff_norm_overflow();
    printf("\tPassed Vector Diff Norm Tests.\n");
}


/**
 * VECTOR SET TESTS
 */
void test_set_four(void) {
    FixedPoint data[4] = { 4389, -132, 389, 83 };
    struct Vector vec = { data, 4 };

    FixedPoint expectedData[4] = { 1, 1, 1, 1 };
    struct Vector expected = { expectedData, 4 };

    vector_set(&vec, 1);

    assert(vector_equal(&expected, &vec));
}


/**
 * VECTOR ADDITION TESTS
 */
void test_add_four(void) {
    FixedPoint data1[4] = { 4389, -132, 389, 83 };
    struct Vector vec1 = { data1, 4 };

    FixedPoint data2[4] = { -92, 3589, 4102, 958 };
    struct Vector vec2 = { data2, 4 };

    FixedPoint expectedData[4] = { 4297, 3457, 4491, 1041 };
    struct Vector expected = { expectedData, 4 };

    vector_add(&vec1, &vec1, &vec2);

    assert(vector_equal(&expected, &vec1));
}


void test_add_ten(void) {
    FixedPoint data1[10] = { 9801, 5014, 6509, 7520, -7067, 3498, -7799, -70, -5553, 4613 };
    struct Vector vec1 = { data1, 10 };

    FixedPoint data2[10] = { 8312, 613, -4558, 5917, -6111, 5522, -6336, 2586, -8420, 2954 };
    struct Vector vec2 = { data2, 10 };

    FixedPoint expectedData[10] = { 18113, 5627, 1951, 13437, -13178, 9020, -14135, 2516, -13973, 7567 };
    struct Vector expected = { expectedData, 10 };

    vector_add(&vec1, &vec1, &vec2);

    assert(vector_equal(&expected, &vec1));
}


/**
 * Vector Multiply Tests
 */
void test_mul_four(void) {
    uint16_t precision = 10;

    FixedPoint data1[4] = { -560,-1751,-586,-1333 };
    struct Vector vec1 = { data1, 4 };

    FixedPoint data2[4] = { -1284,-214,-567,1255 };
    struct Vector vec2 = { data2, 4 };

    FixedPoint expectedData[4] = { 702,365,324,-1634 };
    struct Vector expected = { expectedData, 4 };

    vector_mul(&vec1, &vec1, &vec2, precision);

    assert(vector_equal(&expected, &vec1));
}

 
void test_mul_ten(void) {
    uint16_t precision = 8;

    FixedPoint data1[10] = { 1463,-350,1790,-962,-1646,219,-1830,-749,1935,342 };
    struct Vector vec1 = { data1, 10 };

    FixedPoint data2[10] = { -2006,-1677,-302,1941,-1118,-1191,-1351,-954,1565,-397 };
    struct Vector vec2 = { data2, 10 };

    FixedPoint expectedData[10] = { -11464,2292,-2112,-7294,7188,-1019,9657,2791,11829,-531 };
    struct Vector expected = { expectedData, 10 };

    vector_mul(&vec1, &vec1, &vec2, precision);

    assert(vector_equal(&expected, &vec1));
}


/**
 * Vector Gated Add Tests
 */
void test_gated_add_four(void) {
    uint16_t precision = 10;

    FixedPoint data1[4] = { 513,-1660,835 };
    struct Vector vec1 = { data1, 4 };

    FixedPoint data2[4] = { 1555,1880,1644 };
    struct Vector vec2 = { data2, 4 };

    FixedPoint expectedData[4] = { 1294,995,1441 };
    struct Vector expected = { expectedData, 4 };

    FixedPoint gate = 256;

    vector_gated_add(&vec1, &vec1, &vec2, gate, precision);

    assert(vector_equal(&expected, &vec1));
}


void test_gated_add_ten(void) {
    uint16_t precision = 8;

    FixedPoint data1[10] = { -909,1561,363,931,-258,-241,1077,-173,1665,839 };
    struct Vector vec1 = { data1, 10 };

    FixedPoint data2[10] = { 1059,-806,-1976,1132,-1428,-1508,1501,1689,-1672,612 };
    struct Vector vec2 = { data2, 10 };

    FixedPoint expectedData[10] = { -418,968,-222,981,-551,-558,1182,292,830,782 };
    struct Vector expected = { expectedData, 10 };

    FixedPoint gate = 192;

    vector_gated_add(&vec1, &vec1, &vec2, gate, precision);

    assert(vector_equal(&expected, &vec1));
}

/**
 * VECTOR NORM TESTS
 */
void test_norm_four(void) {
    FixedPoint data[4] = { 709,600,-1899,-737 };
    struct Vector vec = { data, 4 };

    FixedPoint expected = 3945;
    FixedPoint result = vector_norm(&vec);

    assert(result == expected);
}


void test_norm_ten(void) {
    FixedPoint data[10] = { 216,1001,78,248,1289,1522,-563,384,801,-1541 };

    struct Vector vec = { data, 10 };

    FixedPoint expected = 7643;
    FixedPoint result = vector_norm(&vec);

    assert(result == expected);
}

void test_norm_overflow(void) {
    FixedPoint data[9] = { 216,-1001,-32078,248,1289,1522,-563,-384,801 };

    struct Vector vec = { data, 9 };

    FixedPoint expected = INT16_MAX;
    FixedPoint result = vector_norm(&vec);

    assert(result == expected);
}



/**
 * Vector Difference Norm Tests
 */
void test_diff_norm_four(void) {
    FixedPoint data1[4] = { 709,600,-1899,-737 };
    struct Vector vec1 = { data1, 4 };

    FixedPoint data2[4] = { 1428,-1688,1910,334 };
    struct Vector vec2 = { data2, 4 };

    FixedPoint expected = 7887;
    FixedPoint result = vector_diff_norm(&vec1, &vec2);

    assert(result == expected);
}

void test_diff_norm_ten(void) {
    FixedPoint data1[10] = { 216,1001,78,248,1289,1522,-563,384,801,-1541 };
    struct Vector vec1 = { data1, 10 };

    FixedPoint data2[10] = { -818,1215,273,432,2006,-1785,-1505,-608,733,-1395 };
    struct Vector vec2 = { data2, 10 };

    FixedPoint expected = 7799;
    FixedPoint result = vector_diff_norm(&vec1, &vec2);

    assert(result == expected);
}


void test_diff_norm_overflow(void) {
    FixedPoint data1[4] = { 32000,600,-1899,-737 };
    struct Vector vec1 = { data1, 4 };

    FixedPoint data2[4] = { 1428,-1688,1910,334 };
    struct Vector vec2 = { data2, 4 };

    FixedPoint expected = INT16_MAX;
    FixedPoint result = vector_diff_norm(&vec1, &vec2);

    assert(result == expected);
}

/**
 * VECTOR ABSOLUTE DIFFERENCE FUNCTIONS
 */
void test_absolute_diff_four(void) {
    FixedPoint data1[4] = { -1513,146,-1543,295 };
    struct Vector vec1 = { data1, 4 };

    FixedPoint data2[4] = { 586,465,1030,1780 };
    struct Vector vec2 = { data2, 4 };

    FixedPoint expectedData[4] = { 2099,319,2573,1485 };
    struct Vector expected = { expectedData, 4 };

    vector_absolute_diff(&vec1, &vec1, &vec2);
    assert(vector_equal(&expected, &vec1));
}


void test_absolute_diff_ten(void) {
    FixedPoint data1[10] = { -986,-1501,950,519,-1392,1421,-149,1821,-1328,-193 };
    struct Vector vec1 = { data1, 10 };

    FixedPoint data2[10] = { 25,472,1045,1801,351,992,1951,-1316,95,759 };
    struct Vector vec2 = { data2, 10 };

    FixedPoint expectedData[10] = { 1011,1973,95,1282,1743,429,2100,3137,1423,952 };
    struct Vector expected = { expectedData, 10 };

    vector_absolute_diff(&vec1, &vec1, &vec2);
    assert(vector_equal(&expected, &vec1));
}



uint8_t vector_equal(struct Vector *expected, struct Vector *given) {
    if (expected->size != given->size) {
        return 0;
    }

    uint16_t i;
    for (i = 0; i < expected->size; i++) {
        if (expected->data[i] != given->data[i]) {
            return 0;
        }
    }

    return 1;
}

