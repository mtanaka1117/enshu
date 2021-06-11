#include "encoding.h"


uint16_t encode_standard(uint8_t *output, struct Vector *featureVectors, struct BitMap *collectedIndices, uint16_t numFeatures, uint16_t seqLength) {

    // Write the collected index bitmap to the output buffer
    uint16_t outputIdx = 0;

    uint16_t i;
    uint16_t numBytes = collectedIndices->numBytes;
    for (i = numBytes; i > 0; i--) {
        output[outputIdx] = collectedIndices->bytes[outputIdx];
        outputIdx++;
    }

    // Write the encoded features to the output array in 16 bit quantities
    uint16_t seqIdx = 0;  // Assumes that we always collect the first sequence element
    uint16_t byteIdx = 0;
    uint16_t bitIdx = 0;
    uint8_t shouldWrap = 0;
    uint8_t currentBit = 0;
    uint8_t currentByte = 0;

    struct Vector currentVector;
    uint16_t value;

    while (seqIdx < seqLength) {
       
        currentVector = featureVectors[seqIdx];

        for (i = 0; i < numFeatures; i++) {
            value = (uint16_t) (((int32_t) currentVector.data[i]) + TWO_BYTE_OFFSET);

            output[outputIdx] = (value & BYTE_MASK);
            output[outputIdx + 1] = (value >> BITS_PER_BYTE) & BYTE_MASK;
            outputIdx += 2;
        }

        // Advance the featureIdx to the next collected index
        do {
            bitIdx++;
            shouldWrap = bitIdx >= BITS_PER_BYTE;
            bitIdx = bitIdx * (1 - shouldWrap);

            byteIdx += shouldWrap;

            currentByte = collectedIndices->bytes[byteIdx];
            currentBit = (currentByte >> bitIdx) & 1;

            seqIdx++; 
        } while ((seqIdx < seqLength) && !currentBit);
    }

    return outputIdx;
}
