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


uint16_t encode_group(uint8_t *output, struct Vector *featureVectors, struct BitMap *collectedIndices, uint16_t numFeatures, uint16_t seqLength) {

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


uint16_t get_num_bytes(uint16_t numBits) {
    uint8_t remainder = (numBits & 0x7) > 0;
    uint16_t div = numBits >> 3;
    return div + remainder;
}


uint16_t min(uint16_t x, uint16_t y) {
    uint8_t comp = x <= y;
    return (comp * x) + ((1 - comp) * y);
}


void set_group_widths(uint8_t *result, uint16_t *groupSizes, uint8_t numGroups, uint16_t targetBytes, uint16_t startWidth) {
    uint16_t targetBits = targetBytes << 3;

    uint16_t consumedBytes = 0;
    uint16_t i;
    for (i = 0; i < numGroups; i++) {
        consumedBytes += get_num_bytes(groupSizes[i] * startWidth);
        result[i] = startWidth;
    }

    uint16_t size, candidateBytes, diff, width;
    uint16_t counter = 1000;
    uint8_t hasImproved = 1;

    while (hasImproved && (counter > 0)) {
        hasImproved = 0;

        for (i = 0; i < numGroups; i++) {
            // Get the current group width
            width = result[i];
            if (width >= MAX_WIDTH) {
                continue;
            }

            size = groupSizes[i];
            diff = get_num_bytes(size * (width + 1)) - get_num_bytes(size * width);
            candidateBytes = consumedBytes + diff;

            if (candidateBytes <= targetBytes) {
                consumedBytes = candidateBytes;
                hasImproved = 1;
                result[i] = width + 1;
            }
        }

        counter--;
    }

}



uint16_t calculate_grouped_size(uint8_t *groupWidths, uint16_t numCollected, uint16_t numFeatures, uint16_t seqLength, uint16_t groupSize, uint16_t numGroups, uint8_t isBlock) {
    uint16_t totalBytes = 0;
    uint16_t numSoFar = 0;
    uint16_t totalFeatures = numCollected * numFeatures;

    uint16_t numElements, numBits, numBytes;

    uint16_t i;
    for (i = 0; i < numGroups; i++) {
        numElements = min(groupSize, totalFeatures - numSoFar);
        numBits = groupWidths[i] * numElements;
        numBytes = get_num_bytes(numBits);

        totalBytes += numBytes;
        totalFeatures += numElements;
    }

    // Include the meta-data and the sequence mask
    totalBytes += numGroups + get_num_bytes(seqLength) + 1;

    if (isBlock) {
        return AES_BLOCK_SIZE + round_to_aes_block(totalBytes);        
    } else {
        return CHACHA_NONCE_LEN + totalBytes;
    }
}

