#include "encoding_tests.h"


int main(void) {
    printf("==== Test Collected Indices ====\n");
    test_collected_indices_three();
    test_collected_indices_ten();
    test_collected_indices_23();
    test_collected_indices_ten_unset();
    printf("\tPassed Collected Indices Tests.\n");

    printf("==== Test Standard Encoding ====\n");
    test_standard_encode_four();
    test_standard_encode_ten();
    printf("\tPassed Standard Encoding Tests.\n");

    printf("==== Test Block Rounding ====\n");
    test_rounding_equal();
    test_rounding_five();
    test_rounding_ten();
    printf("\tPassed Block Rounding Tests.\n");

    printf("==== Test Group Message Length Estimation ====\n");
    test_group_length_block_1();
    printf("\tPassed Group Message Length Estimation.\n");

    printf("==== Test Group Width Setting ====\n");
    test_set_widths_3();
    test_set_widths_4();
    test_set_widths_5();
    printf("\tPassed Group Width Setting.\n");

    printf("==== Testing Measurement Pruning ====\n");
    test_pruning_4_1();
    test_pruning_4_2();
    printf("\tPassed Pruning Tests.\n");

    printf("==== Testing Group Encoding ====\n");
    test_group_encode_four();
    printf("\tPassed Group Encoding Tests.\n");

    return 0;
}

void test_collected_indices_three() {
    uint8_t buffer[1];
    struct BitMap bitmap = { buffer, 1 };
    clear_bitmap(&bitmap);

    uint16_t numCollected = 2;
    uint16_t collectedIndices[2] = { 0, 2 };

    uint16_t i = 0;
    for (; i < numCollected; i++) {
        set_bit(collectedIndices[i], &bitmap);
    }

    assert(bitmap.numBytes == 1);
    assert(bitmap.bytes[0] == 5);
}


void test_collected_indices_ten() {
    uint8_t buffer[2];
    struct BitMap bitmap = { buffer, 2 };
    clear_bitmap(&bitmap);

    uint16_t numCollected = 5;
    uint16_t collectedIndices[5] = { 0, 4, 5, 8, 9 };

    uint16_t i = 0;
    for (; i < numCollected; i++) {
        set_bit(collectedIndices[i], &bitmap);
    }

    assert(bitmap.numBytes == 2);
    assert(bitmap.bytes[0] == 49);
    assert(bitmap.bytes[1] == 3);
}


void test_collected_indices_23() {
    uint8_t buffer[3];
    struct BitMap bitmap = { buffer, 3 };
    clear_bitmap(&bitmap);

    uint16_t numCollected = 12;
    uint16_t collectedIndices[12] = { 0, 4, 5, 8, 9, 12, 15, 17, 18, 20, 21, 22 };

    uint16_t i = 0;
    for (; i < numCollected; i++) {
        set_bit(collectedIndices[i], &bitmap);
    }

    assert(bitmap.numBytes == 3);
    assert(bitmap.bytes[0] == 49);
    assert(bitmap.bytes[1] == 147);
    assert(bitmap.bytes[2] == 118);
}


void test_collected_indices_ten_unset() {
    uint8_t buffer[2];
    struct BitMap bitmap = { buffer, 2 };
    clear_bitmap(&bitmap);

    uint16_t numCollected = 5;
    uint16_t collectedIndices[5] = { 0, 4, 5, 8, 9 };

    uint16_t i = 0;
    for (; i < numCollected; i++) {
        set_bit(collectedIndices[i], &bitmap);
    }

    assert(bitmap.numBytes == 2);
    assert(bitmap.bytes[0] == 49);
    assert(bitmap.bytes[1] == 3);

    unset_bit(collectedIndices[4], &bitmap);
    assert(bitmap.bytes[1] == 0x01);

    unset_bit(collectedIndices[1], &bitmap);
    assert(bitmap.bytes[0] == 0x21);
}


void test_standard_encode_four() {
    uint16_t seqLength = 4;
    uint16_t numFeatures = 3;

    uint8_t buffer[1];
    struct BitMap bitmap = { buffer, 1 };
    clear_bitmap(&bitmap);

    uint16_t numCollected = 2;
    uint16_t collectedIndices[2] = { 0, 2 };

    uint16_t i = 0;
    for (; i < numCollected; i++) {
        set_bit(collectedIndices[i], &bitmap);
    }

    // 3 features per vector, 2 collected vectors
    struct Vector collectedFeatures[4];
    FixedPoint features[6] = { 23111, -3040, -28386, 5672, -4823, -32679 };

    collectedFeatures[0].data = features;
    collectedFeatures[0].size = numFeatures;

    collectedFeatures[2].data = features + numFeatures;
    collectedFeatures[2].size = numFeatures;

    uint8_t encoded[13];
    uint16_t numBytes = encode_standard(encoded, collectedFeatures, &bitmap, numFeatures, seqLength);

    uint8_t expected[13] = { 0x05, 0x47, 0xDA, 0x20, 0x74, 0x1E, 0x11, 0x28, 0x96, 0x29, 0x6D, 0x59, 0x00 };

    assert(numBytes == 13);

    for (uint16_t i = 0; i < 13; i++) {
        assert(encoded[i] == expected[i]);
    }
}


void test_standard_encode_ten() {
    uint16_t seqLength = 10;
    uint16_t numFeatures = 3;

    uint8_t buffer[2];
    struct BitMap bitmap = { buffer, 2 };
    clear_bitmap(&bitmap);

    uint16_t numCollected = 2;
    uint16_t collectedIndices[2] = { 0, 8 };

    uint16_t i = 0;
    for (; i < numCollected; i++) {
        set_bit(collectedIndices[i], &bitmap);
    }

    // 3 features per vector, 2 collected vectors
    struct Vector collectedFeatures[10];
    FixedPoint features[6] = { 23111, -3040, -28386, 5672, -4823, -32679 };

    collectedFeatures[0].data = features;
    collectedFeatures[0].size = numFeatures;

    collectedFeatures[8].data = features + numFeatures;
    collectedFeatures[8].size = numFeatures;

    uint8_t encoded[14];
    uint16_t numBytes = encode_standard(encoded, collectedFeatures, &bitmap, numFeatures, seqLength);

    uint8_t expected[14] = { 0x01, 0x01, 0x47, 0xDA, 0x20, 0x74, 0x1E, 0x11, 0x28, 0x96, 0x29, 0x6D, 0x59, 0x00 };

    assert(numBytes == 14);

    for (uint16_t i = 0; i < 14; i++) {
        assert(encoded[i] == expected[i]);
    }
}


void test_rounding_equal(void) {
    assert(round_to_aes_block(AES_BLOCK_SIZE) == AES_BLOCK_SIZE);
    assert(round_to_aes_block(2 * AES_BLOCK_SIZE) == (2 * AES_BLOCK_SIZE));
    assert(round_to_aes_block(7 * AES_BLOCK_SIZE) == (7 * AES_BLOCK_SIZE));
}


void test_rounding_five(void) {
    assert(round_to_aes_block(5) == AES_BLOCK_SIZE);
    assert(round_to_aes_block(25) == (2 * AES_BLOCK_SIZE));
    assert(round_to_aes_block(105) == (7 * AES_BLOCK_SIZE));
}


void test_rounding_ten(void) {
    assert(round_to_aes_block(10) == AES_BLOCK_SIZE);
    assert(round_to_aes_block(30) == (2 * AES_BLOCK_SIZE));
    assert(round_to_aes_block(110) == (7 * AES_BLOCK_SIZE));
}


void test_group_length_block_1(void) {
    uint8_t groupWidths[2] = { 6, 7 };
    uint16_t numGroups = 2;

    uint16_t length = calculate_grouped_size(groupWidths, 4, 3, 10, 6, numGroups, 1);
    assert(length == 32);
}


void test_set_widths_3(void) {
    uint16_t groupSizes[3] = { 20, 12, 15 };
    uint8_t numGroups = 3;
    uint16_t targetBytes = 39;
    uint16_t startWidth = 5;

    uint8_t result[3];
    set_group_widths(result, groupSizes, numGroups, targetBytes, startWidth);

    assert(result[0] == 7);
    assert(result[1] == 6);
    assert(result[2] == 6);
}


void test_set_widths_4(void) {
    uint16_t groupSizes[4] = { 150, 12, 15, 7 };
    uint8_t numGroups = 4;
    uint16_t targetBytes = 129;
    uint16_t startWidth = 5;

    uint8_t result[4];
    set_group_widths(result, groupSizes, numGroups, targetBytes, startWidth);

    assert(result[0] == 5);
    assert(result[1] == 8);
    assert(result[2] == 8);
    assert(result[3] == 9);
}


void test_set_widths_5(void) {
    uint16_t groupSizes[5] = { 8, 150, 12, 15, 7 };
    uint8_t numGroups = 5;
    uint16_t targetBytes = 137;
    uint16_t startWidth = 5;

    uint8_t result[5];
    set_group_widths(result, groupSizes, numGroups, targetBytes, startWidth);

    assert(result[0] == 9);
    assert(result[1] == 5);
    assert(result[2] == 8);
    assert(result[3] == 8);
    assert(result[4] == 8);
}


void test_pruning_4_1(void) {
    uint16_t seqLength = 6;
    struct Vector measurements[6];

    uint16_t precision = 10;
    FixedPoint one = 1 << precision;
    FixedPoint two = 1 << (precision + 1);
    FixedPoint half = 1 << (precision - 1);

    FixedPoint features[12] = { one, two, -one, half, 0, 0, -one - half, one, 0, 0, two, one };
    uint16_t i;
    for (i = 0; i < seqLength; i++) {
        measurements[i].size = 2;
        measurements[i].data = features + (i * 2);
    }

    uint8_t collectedBuffer[1] = { 0 };
    struct BitMap collectedIdx = { collectedBuffer, 1 };
    set_bit(0, &collectedIdx);
    set_bit(1, &collectedIdx);
    set_bit(3, &collectedIdx);
    set_bit(5, &collectedIdx);

    assert(collectedIdx.bytes[0] == 0x2B);
    prune_sequence(measurements, &collectedIdx, 4, 3, seqLength, precision);
    assert(collectedIdx.bytes[0] == 0x23);
}


void test_pruning_4_2(void) {
    uint16_t seqLength = 6;
    struct Vector measurements[6];

    uint16_t precision = 10;
    FixedPoint one = 1 << precision;
    FixedPoint two = 1 << (precision + 1);
    FixedPoint half = 1 << (precision - 1);

    FixedPoint features[12] = { one, two, -one, half, 0, 0, -one - half, one, 0, 0, two, one };
    uint16_t i;
    for (i = 0; i < seqLength; i++) {
        measurements[i].size = 2;
        measurements[i].data = features + (i * 2);
    }

    uint8_t collectedBuffer[1] = { 0 };
    struct BitMap collectedIdx = { collectedBuffer, 1 };
    set_bit(0, &collectedIdx);
    set_bit(1, &collectedIdx);
    set_bit(3, &collectedIdx);
    set_bit(5, &collectedIdx);

    assert(collectedIdx.bytes[0] == 0x2B);
    prune_sequence(measurements, &collectedIdx, 4, 2, seqLength, precision);
    assert(collectedIdx.bytes[0] == 0x21);
}



void test_group_encode_four() {
    uint16_t seqLength = 4;
    uint16_t numFeatures = 3;

    uint8_t buffer[1];
    struct BitMap bitmap = { buffer, 1 };
    clear_bitmap(&bitmap);

    uint16_t numCollected = 2;
    uint16_t collectedIndices[2] = { 0, 2 };

    uint16_t i = 0;
    for (; i < numCollected; i++) {
        set_bit(collectedIndices[i], &bitmap);
    }

    // 3 features per vector, 2 collected vectors
    struct Vector collectedFeatures[4];
    FixedPoint features[6] = { 23111, -3040, -28386, 5672, -4823, -32679 };

    collectedFeatures[0].data = features;
    collectedFeatures[0].size = numFeatures;

    collectedFeatures[2].data = features + numFeatures;
    collectedFeatures[2].size = numFeatures;

    uint16_t targetBytes = 48;
    FixedPoint tempBuffer[12];
    int8_t shiftBuffer[12];

    uint8_t encoded[32];
    uint16_t numBytes = encode_group(encoded, collectedFeatures, &bitmap, numCollected, numFeatures, seqLength, targetBytes, 10, tempBuffer, shiftBuffer, 1);

    uint8_t expected[13] = { 0x05, 0x47, 0xDA, 0x20, 0x74, 0x1E, 0x11, 0x28, 0x96, 0x29, 0x6D, 0x59, 0x00 };

    assert(numBytes == 13);

    for (uint16_t i = 0; i < 13; i++) {
        assert(encoded[i] == expected[i]);
    }
}
