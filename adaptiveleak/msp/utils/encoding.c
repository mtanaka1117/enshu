#include "encoding.h"


uint8_t GROUP_WIDTHS[MAX_NUM_GROUPS];


uint16_t encode_collected_indices(uint8_t *output, struct BitMap *collectedIndices, uint16_t outputIdx) {
    uint16_t i;
    uint16_t numBytes = collectedIndices->numBytes;
    for (i = numBytes; i > 0; i--) {
        output[outputIdx] = collectedIndices->bytes[outputIdx];
        outputIdx++;
    }

    return outputIdx;
}


uint16_t encode_shifts(uint8_t *output, int8_t *shifts, uint8_t *widths, uint16_t *counts, uint8_t countBits, uint8_t numGroups, uint16_t outputIdx) {
    output[outputIdx] = ((numGroups << 4) & 0xF0) | (countBits & 0xF);
    outputIdx++;

    uint16_t packedBytes = pack(output + outputIdx, (int16_t *) counts, countBits, numGroups, 0);
    outputIdx += packedBytes;

    uint8_t i;
    for (i = 0; i < numGroups; i++) {
        output[outputIdx] = ((widths[i] & WIDTH_MASK) << NUM_SHIFT_BITS) | ((shifts[i] + SHIFT_OFFSET) & SHIFT_MASK);
        outputIdx++;
    }

    return outputIdx;
}


uint16_t encode_standard(uint8_t *output, struct Vector *featureVectors, struct BitMap *collectedIndices, uint16_t numFeatures, uint16_t seqLength) {

    // Write the collected index bitmap to the output buffer
    uint16_t outputIdx = 0;
    outputIdx = encode_collected_indices(output, collectedIndices, outputIdx);

    // Write the encoded features to the output array in 16 bit quantities
    uint16_t seqIdx = 0;  // Assumes that we always collect the first sequence element
    uint16_t byteIdx = 0;
    uint16_t bitIdx = 0;
    uint8_t shouldWrap = 0;
    uint8_t currentBit = 0;
    uint8_t currentByte = 0;

    struct Vector currentVector;
    uint16_t value, i;

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


uint16_t encode_group(uint8_t *output,
                      struct Vector *featureVectors,
                      struct BitMap *collectedIndices,
                      uint16_t numCollected,
                      uint16_t numFeatures,
                      uint16_t seqLength,
                      uint16_t sizeBytes,
                      uint16_t targetBytes,
                      uint16_t precision,
                      uint16_t maxCollected,
                      FixedPoint *tempBuffer,
                      int8_t *shiftBuffer,
                      uint16_t *countBuffer,
                      uint8_t isBlock) {
    // Estimate the meta-data associated with stable group encoding
    uint16_t maskBytes = collectedIndices->numBytes;
    uint16_t metadataBytes = maskBytes + sizeBytes + MAX_NUM_GROUPS + 1;

    if (isBlock) {
        metadataBytes += AES_BLOCK_SIZE;
    } else {
        metadataBytes += CHACHA_NONCE_LEN;
    }

    // Compute the target number of data bytes
    uint16_t targetDataBytes = targetBytes - metadataBytes;
    uint16_t targetDataBits = (targetDataBytes - MAX_NUM_GROUPS) << 3;

    // Prune measurements if needed
    if (numCollected > maxCollected) {
        prune_sequence(featureVectors, collectedIndices, numCollected, maxCollected, seqLength, precision);
        numCollected = maxCollected;
    }
    
    // Write the collected features in transpose fashion into the temp buffer
    uint16_t seqIdx = 0;  // Assumes that we always collect the first sequence element
    uint16_t byteIdx = 0;
    uint16_t bitIdx = 0;
    uint8_t shouldWrap = 0;
    uint8_t currentBit = 0;
    uint8_t currentByte = 0;

    struct Vector currentVector;
    uint16_t featureIdx, i;
    uint16_t collectedIdx = 0;

    while (seqIdx < seqLength) {
       
        currentVector = featureVectors[seqIdx];

        // Write the current vector in transpose fashion
        for (i = 0; i < numFeatures; i++) {
            featureIdx = i * numCollected + collectedIdx;
            tempBuffer[featureIdx] = currentVector.data[i];
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

        collectedIdx++;
    }

    // Get the range shifts
    uint16_t count = numCollected * numFeatures;
    uint16_t minWidth = targetDataBits / (numFeatures * numCollected);
    minWidth = min16u(minWidth, MAX_WIDTH);

    get_range_shifts_array(shiftBuffer, tempBuffer, precision, minWidth, NUM_SHIFT_BITS, count);

    // Run-Length Encode the range shifts
    uint16_t numGroups = create_shift_groups(shiftBuffer, countBuffer, shiftBuffer, count, MAX_NUM_GROUPS); 

    // Re-calculate the meta-data size based on the given number of shift groups
    metadataBytes -= sizeBytes;

    uint16_t maxSize = 0;
    uint16_t s;
    for (i = 0; i < numGroups; i++) {
        s = countBuffer[i];
        if (s > maxSize) {
            maxSize = s;
        }
    }

    uint8_t sizeWidth = 0;
    uint16_t tempSize = maxSize;
    while (tempSize > 0) {
        tempSize = tempSize >> 1;
        sizeWidth++;
    }

    sizeBytes = get_num_bytes(sizeWidth * numGroups);
    metadataBytes += sizeBytes;
    targetDataBytes = targetBytes - metadataBytes;

    // Set the group widths
    set_group_widths(GROUP_WIDTHS, countBuffer, numGroups, targetDataBytes, minWidth);

    // Write the collected index bitmap to the output buffer
    uint16_t outputIdx = 0;
    outputIdx = encode_collected_indices(output, collectedIndices, outputIdx);

    // Write the shift and width values
    outputIdx = encode_shifts(output, shiftBuffer, GROUP_WIDTHS, countBuffer, sizeWidth, numGroups, outputIdx);

    // Write the encoded feature values
    uint16_t nonFractional = 16 - precision;
    uint16_t groupSize, groupPrecision, groupWidth, groupShift;
    featureIdx = 0;

    for (i = 0; i < numGroups ; i++) {
        groupSize = countBuffer[i];
        groupWidth = GROUP_WIDTHS[i];
        groupShift = shiftBuffer[i];
        groupPrecision = groupWidth - nonFractional - groupShift;

        if ((featureIdx + groupSize) > count) {
            groupSize = count - featureIdx;
        }

        fp_convert_array(tempBuffer, precision, groupPrecision, groupWidth, featureIdx, groupSize);
        outputIdx += pack(output + outputIdx, tempBuffer + featureIdx, groupWidth, groupSize, 1);

        featureIdx += groupSize;
    }

    return outputIdx;
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
    uint16_t counter = MAX_WIDTH - MIN_WIDTH;
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
        numElements = min16u(groupSize, totalFeatures - numSoFar);
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

