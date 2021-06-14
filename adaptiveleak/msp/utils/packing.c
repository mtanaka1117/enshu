#include "packing.h"


uint16_t min(uint16_t x, uint16_t y) {
    uint8_t cond = x < y;
    return cond * x + (1 - cond) * y;
}


uint16_t pack(uint8_t *output, uint16_t *values, uint8_t bitWidth, uint16_t numValues) {
    /**
     * Packs the given features into the output byte array using the given bit width per feature.
     */
    if (numValues == 0) {
        return 0;
    }

    // Get the number of bytes for each value
    uint8_t bytesPerVal = (bitWidth >> 3);
    uint8_t widthRemainder = bitWidth & 0x7;
    bytesPerVal += (widthRemainder > 0);

    uint16_t current = 0;
    uint16_t currentByte = 0;
    uint16_t value = 0;
    uint16_t consumedBits = 0;
    uint16_t outputIdx = 0;

    uint8_t numBits = 0;
    uint8_t usedBits = 0;

    uint16_t byteIdx, valIdx;

    for (valIdx = 0; valIdx < numValues; valIdx++) {
        value = values[valIdx];

        for (byteIdx = 0; byteIdx < bytesPerVal; byteIdx++) {
            currentByte = (value >> (byteIdx * BITS_PER_BYTE)) & 0xFF;

            if ((!widthRemainder) || (byteIdx < (bytesPerVal - 1))) {
                numBits = BITS_PER_BYTE;
            } else {
                numBits = widthRemainder;
            }

            // Set bits in the current byte
            current |= (currentByte << consumedBits);
            current &= 0xFF;

            // Add to the number of consumed bits in the current value
            usedBits = min(BITS_PER_BYTE - consumedBits, numBits); 
            consumedBits += numBits;

            if (consumedBits > BITS_PER_BYTE) {
                consumedBits -= BITS_PER_BYTE;
                output[outputIdx++] = current;
                current = currentByte >> usedBits;
            }
        }
    }

    output[outputIdx++] = current;

    return outputIdx;
}

