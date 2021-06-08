#include "encoding_tests.h"


int main(void) {
    printf("==== Test Collected Indices ====\n");
    test_collected_indices_three();
    test_collected_indices_ten();
    test_collected_indices_23();
    printf("\tPassed Collected Indices Tests.\n");

    printf("==== Test Standard Encoding ====\n");
    test_standard_encode_three();
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


void test_standard_encode_three() {
    uint8_t buffer[1];
    struct BitMap bitmap = { buffer, 1 };
    clear_bitmap(&bitmap);

    uint16_t numCollected = 2;
    uint16_t collectedIndices[2] = { 0, 2 };

    uint16_t i = 0;
    for (; i < numCollected; i++) {
        set_bit(collectedIndices[i], &bitmap);
    }

    uint16_t numFeatures = 6;
    FixedPoint features[6] = { 23111, -3040, -28386, 5672, -4823, -32679 };

    uint8_t encoded[13];
    uint16_t numBytes = encode_standard(encoded, features, &bitmap, numFeatures);

    uint8_t expected[13] = { 0x05, 0xDA, 0x46, 0x74, 0x1F, 0x11, 0x1D, 0x96, 0x27, 0x6D, 0x28, 0x00, 0x58 };

    assert(numBytes == 13);

    for (uint16_t i = 0; i < 13; i++) {
        assert(encoded[i] == expected[i]);
    }
}
