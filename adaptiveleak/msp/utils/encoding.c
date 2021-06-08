#include "encoding.h"


uint16_t encode_standard(uint8_t *output, FixedPoint *features, struct BitMap *collectedIndices, uint16_t numFeatures) {

    // Write the collected index bitmap to the output buffer
    uint16_t outputIdx = 0;

    uint16_t i;
    uint16_t numBytes = collectedIndices->numBytes;
    for (i = numBytes; i > 0; i--) {
        output[outputIdx] = collectedIndices->bytes[outputIdx];
        outputIdx++;
    }

    // Write the encoded features to the output array in 16 bit quantities
    uint16_t featureIdx = 0;
    uint16_t value;

    for (i = numFeatures; i > 0; i--) {
        value = (uint16_t) (((int32_t) features[featureIdx]) + TWO_BYTE_OFFSET);

        output[outputIdx + 1] = (value & BYTE_MASK);
        output[outputIdx] = (value >> BITS_PER_BYTE) & BYTE_MASK;

        featureIdx++;
        outputIdx += 2;
    }

    return outputIdx;
}
