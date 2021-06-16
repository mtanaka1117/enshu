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
    test_gated_add_three();
    test_gated_add_ten();
    printf("\tPassed Vector Gated Add Tests.\n");

    printf("==== Testing Vector-Scalar Gated Add ====\n");
    test_gated_add_scalar_three();
    test_gated_add_scalar_ten();
    printf("\tPassed Vector-Scaler Gated Add Tests.\n");

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

    printf("==== Testing Matrix Vector Products ====\n");
    test_mat_vec_prod_3_4();
    test_mat_vec_prod_6_5();
    printf("\tPassed Matrix Vector Products Tests.\n");

    printf("==== Testing Matrix Vector Products ====\n");
    test_dot_prod_4();
    test_dot_prod_6();
    printf("\tPassed Matrix Vector Products Tests.\n");

    printf("==== Testing Vector Stacking ====\n");
    test_stack_1_10();
    test_stack_6_6();
    printf("\tPassed Vector Stacking.\n");

    printf("==== Testing Vector Scaling ====\n");
    test_scale_6();
    test_scale_6_down();
    test_scale_10_up();
    printf("\tPassed Vector Scaling.\n");

    printf("==== Testing Vector Apply ====\n");
    test_apply_tanh();
    test_apply_sigmoid();
    printf("\tPassed Vector Apply.\n");
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
 * VECTOR GATED ADD TESTS
 */
void test_gated_add_three(void) {
    uint16_t precision = 10;

    FixedPoint data1[3] = { 513,-1660,835 };
    struct Vector vec1 = { data1, 3 };

    FixedPoint data2[3] = { 1555,1880,1644 };
    struct Vector vec2 = { data2, 3 };

    FixedPoint gateData[3] = { 832,993,17 };
    struct Vector gate = { gateData, 3 };

    FixedPoint expectedData[3] = { 707,-1554,1629 };
    struct Vector expected = { expectedData, 3 };

    vector_gated_add(&vec1, &vec1, &vec2, &gate, precision);

    assert(vector_equal(&expected, &vec1));
}


void test_gated_add_ten(void) {
    uint16_t precision = 9;

    FixedPoint data1[10] = { 1342,-3268,1750,1870,4524,-1686,-1815,2724,-4046,-2958 };
    struct Vector vec1 = { data1, 10 };

    FixedPoint data2[10] = { -545,565,1286,3642,-3893,2573,4847,2785,1439,-3208 };
    struct Vector vec2 = { data2, 10 };

    FixedPoint gateData[10] = { 207,373,84,139,171,362,207,344,286,306 };
    struct Vector gate = { gateData, 10 };

    FixedPoint expectedData[10] = { 217,-2228,1362,3160,-1083,-440,2153,2743,-1626,-3059 };
    struct Vector expected = { expectedData, 10 };

    vector_gated_add(&vec1, &vec1, &vec2, &gate, precision);

    assert(vector_equal(&expected, &vec1));
}


/**
 * VECTOR GATES SCALAR ADD TESTS
 */
void test_gated_add_scalar_three(void) {
    uint16_t precision = 10;

    FixedPoint data1[3] = { 513,-1660,835 };
    struct Vector vec1 = { data1, 3 };

    FixedPoint data2[3] = { 1555,1880,1644 };
    struct Vector vec2 = { data2, 3 };

    FixedPoint expectedData[3] = { 1294,995,1441 };
    struct Vector expected = { expectedData, 3 };

    FixedPoint gate = 256;

    vector_gated_add_scalar(&vec1, &vec1, &vec2, gate, precision);

    assert(vector_equal(&expected, &vec1));
}


void test_gated_add_scalar_ten(void) {
    uint16_t precision = 8;

    FixedPoint data1[10] = { -909,1561,363,931,-258,-241,1077,-173,1665,839 };
    struct Vector vec1 = { data1, 10 };

    FixedPoint data2[10] = { 1059,-806,-1976,1132,-1428,-1508,1501,1689,-1672,612 };
    struct Vector vec2 = { data2, 10 };

    FixedPoint expectedData[10] = { -418,968,-222,981,-551,-558,1182,292,830,782 };
    struct Vector expected = { expectedData, 10 };

    FixedPoint gate = 192;

    vector_gated_add_scalar(&vec1, &vec1, &vec2, gate, precision);

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

/**
 * MATRIX VECTOR PRODUCT TESTS
 */
void test_mat_vec_prod_3_4(void) {
    FixedPoint vecData[4] = { -838,-3669,2469,4650 };
    struct Vector vec = { vecData, 4 }; 

    FixedPoint matData[12] = { 1647,2710,-1735,-3532,706,-2512,-2117,3306,81,1143,-1358,-4733 };
    struct Matrix matrix = { matData, 4, 3 };    
 
    FixedPoint expectedData[3] = { 11392,-2944,-10879 };
    struct Vector expected = { expectedData, 3 };

    FixedPoint resultData[3];
    struct Vector result = { resultData, 3 };

    matrix_vector_prod(&result, &matrix, &vec, 10);
    assert(vector_equal(&expected, &result));
}

void test_mat_vec_prod_6_5(void) {
    FixedPoint vecData[5] = { 1427,-550,23,-753,-4619 };
    struct Vector vec = { vecData, 5 };

    FixedPoint matData[30] = { 524,-2263,3261,4050,3494,-1901,-524,-2004,4108,-1761,2714,-1095,311,-4356,4028,-1030,510,202,-169,4853,-3662,-4349,2569,-523,-4478,-4031,-4768,-100,4860,-139 };
    struct Matrix matrix = { matData, 5, 6 };    
 
    FixedPoint expectedData[6] = { 21340,12437,26626,10213,-20391,-1048 };
    struct Vector expected = { expectedData, 6 };

    FixedPoint resultData[6];
    struct Vector result = { resultData, 6 };

    matrix_vector_prod(&result, &matrix, &vec, 10);
    assert(vector_equal(&expected, &result));
}


/**
 * VECTOR DOT PRODUCT TESTS
 */
void test_dot_prod_4(void) {
    FixedPoint vec1Data[4] = { -3423,4707,2468,-3056 };
    struct Vector vec1 = { vec1Data, 4 };

    FixedPoint vec2Data[4] = { 2466,-4430,1701,278 };
    struct Vector vec2 = { vec2Data, 4 };

    FixedPoint result = vector_dot_prod(&vec1, &vec2, 10);
    assert(result == -25339);
}


void test_dot_prod_6(void) {
    FixedPoint vec1Data[6] = { 999,-920,1778,1284,-881,-841 };
    struct Vector vec1 = { vec1Data, 6 };

    FixedPoint vec2Data[6] ={ -2519,-878,-1434,4544,-2751,-4086 };
    struct Vector vec2 = { vec2Data, 6 };

    FixedPoint result = vector_dot_prod(&vec1, &vec2, 11);
    assert(result == 3628);
}



/**
 * VECTOR STACKING TESTS
 */
void test_stack_1_10(void) {
    FixedPoint vec1Data[1] = { 2315 };
    struct Vector vec1 = { vec1Data, 1 };

    FixedPoint vec2Data[10] = { -2046,-1244,3895,4922,-2941,-3425,-665,1555,-2198,-3269 };
    struct Vector vec2 = { vec2Data, 10 };

    FixedPoint resultData[11];
    struct Vector result = { resultData, 11 };

    FixedPoint expectedData[11] = { 2315,-2046,-1244,3895,4922,-2941,-3425,-665,1555,-2198,-3269 };
    struct Vector expected = { expectedData, 11 };

    vector_stack(&result, &vec1, &vec2);
    assert(vector_equal(&expected, &result));
}


void test_stack_6_6(void) {
    FixedPoint vec1Data[6] = { 2315,-2046,-1244,3895,4922,-2941 };
    struct Vector vec1 = { vec1Data, 6 };

    FixedPoint vec2Data[6] = { -3425,-665,1555,-2198,-3269,35 };
    struct Vector vec2 = { vec2Data, 6 };

    FixedPoint resultData[12];
    struct Vector result = { resultData, 12 };

    FixedPoint expectedData[12] = { 2315,-2046,-1244,3895,4922,-2941,-3425,-665,1555,-2198,-3269,35 };
    struct Vector expected = { expectedData, 12 };

    vector_stack(&result, &vec1, &vec2);
    assert(vector_equal(&expected, &result));
}

/**
 * VECTOR SCALING TESTS
 */
void test_scale_6(void) {
    FixedPoint vecData[6] = { -3995,619,-883,-4957,3213,3183 };
    struct Vector vec = { vecData, 6 };

    FixedPoint meanData[6] = { 2675,4915,-802,2527,-4585,1745 };
    struct Vector mean = { meanData, 6 };

    FixedPoint scaleData[6] = { 2214,4593,878,4964,3618,-1508 };
    struct Vector scale = { scaleData, 6 };

    FixedPoint expectedData[6] = { -7211,-9635,-35,-18140,13775,-1059 };
    struct Vector expected = { expectedData, 6 };

    vector_scale(&vec, &vec, &mean, &scale, 11, 11);
    assert(vector_equal(&expected, &vec));
}

void test_scale_6_down(void) {
    FixedPoint vecData[6] = { -3995,619,-883,-4957,3213,3183 };
    struct Vector vec = { vecData, 6 };

    FixedPoint meanData[6] = { 2675,4915,-802,2527,-4585,1745 };
    struct Vector mean = { meanData, 6 };

    FixedPoint scaleData[6] = { 2214,4593,878,4964,3618,-1508 };
    struct Vector scale = { scaleData, 6 };

    FixedPoint expectedData[6] = { -1803,-2409,-9,-4535,3443,-265 };
    struct Vector expected = { expectedData, 6 };

    vector_scale(&vec, &vec, &mean, &scale, 11, 9);
    assert(vector_equal(&expected, &vec));
}


void test_scale_10_up(void) {
    FixedPoint vecData[10] = { 910,425,-292,718,-712,-599,126,-134,-502,-214 };
    struct Vector vec = { vecData, 10 };

    FixedPoint meanData[10] = { -58,-228,-370,459,575,63,-683,-66,-941,-428 };
    struct Vector mean = { meanData, 10 };

    FixedPoint scaleData[10] = { 585,218,419,53,350,68,692,132,877,679 };
    struct Vector scale = { scaleData, 10 };

    FixedPoint expectedData[10] = { 8848,2224,504,208,-7040,-704,8744,-144,6008,2264 };
    struct Vector expected = { expectedData, 10 };

    vector_scale(&vec, &vec, &mean, &scale, 9, 12);
    assert(vector_equal(&expected, &vec));
}


/**
 * VECTOR APPLY TESTS
 */
void test_apply_tanh(void) {
    uint16_t precision = 10;
        
    FixedPoint vecData[6] = { 0,512,-512,1024,2040,-4000 };
    struct Vector vec = { vecData, 6 };

    FixedPoint expectedData[6] = { 0,464,-464,768,991,-1024 };
    struct Vector expected = { expectedData, 6 };

    vector_apply(&vec, &vec, &fp_tanh, precision);
    assert(vector_equal(&expected, &vec));
}


void test_apply_sigmoid(void) {
    uint16_t precision = 10;
        
    FixedPoint vecData[7] = { 0,512,-512,1024,2040,5000,-5000 };
    struct Vector vec = { vecData, 7 };

    FixedPoint expectedData[7] = { 512,640,384,744,895,1024,0 };
    struct Vector expected = { expectedData, 7 };

    vector_apply(&vec, &vec, &fp_sigmoid, precision);
    assert(vector_equal(&expected, &vec));
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

