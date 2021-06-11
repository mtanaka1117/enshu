#include "encoding_tests.h"


int main(void) {
    printf("==== Test Collected Indices ====\n");
    test_collected_indices_three();
    test_collected_indices_ten();
    test_collected_indices_23();
    printf("\tPassed Collected Indices Tests.\n");

    printf("==== Test Standard Encoding ====\n");
    test_standard_encode_four();
    test_standard_encode_ten();
    printf("\tPassed Standard Encoding Tests.\n");
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
