#include "pruning.h"


#define MAX_PRUNING_SIZE 30
uint16_t LOWEST_SCORES[MAX_PRUNING_SIZE];
uint16_t LOWEST_IDX[MAX_PRUNING_SIZE];


uint16_t max(uint16_t x, uint16_t y) {
    uint8_t comp = (x < y);
    return (comp * y) + ((1 - comp) * x);
}


uint16_t insertSorted(uint16_t score, uint16_t idx, uint16_t *lowestScores, uint16_t *lowestIdx, uint16_t currentMax, uint16_t *currentLength, uint16_t maxLength) {
    uint16_t length = *currentLength;

    if (length < maxLength) {
        lowestScores[length] = score;
        lowestIdx[length] = idx;
        *currentLength += 1;
    } else {
        // Find the maximum element
        uint16_t maxScore = 0;
        uint16_t maxIdx = 0;

        uint16_t i, score;
        for (i = 0; i < maxLength; i++) {
            score = lowestScores[i];
            if (score > maxScore) {
                maxScore = score;
                maxIdx = i;
            }
        }

        lowestScores[maxIdx] = score;
        lowestIdx[maxIdx] = idx;
    }

    return max(score, currentMax);
}




void prune_sequence(struct Vector *measurements, struct BitMap *collectedIndices, uint16_t numCollected, uint16_t maxCollected, uint16_t seqLength, uint16_t precision) {
    // Iterate through the collected indices to compute the scores for each feature vector
    uint16_t seqIdx = 0;  // Assumes that we always collect the first sequence element
    uint16_t prevIdx = 0;
    uint16_t byteIdx = 0;
    uint16_t bitIdx = 0;
    uint8_t shouldWrap = 0;
    uint8_t currentBit = 0;
    uint8_t currentByte = 0;

    volatile FixedPoint norm;
    volatile uint16_t idxDiff;
    volatile uint16_t score;

    uint16_t numToPrune = numCollected - maxCollected;
    uint16_t currentLength = 0;
    uint16_t currentMax = 0;

    while (seqIdx < seqLength) {
        if (seqIdx > 0) {
            norm = vector_diff_norm(measurements + seqIdx, measurements + prevIdx);
            norm = (norm >> precision);  // Convert out of fixed point for numerical stability
            idxDiff = (seqIdx - prevIdx) >> 3;
            score = norm + idxDiff;

            if (score < currentMax) {
                insertSorted(score, seqIdx, LOWEST_SCORES, LOWEST_IDX, currentMax, &currentLength, numToPrune);
            }
        }

        prevIdx = seqIdx;

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

    // Un-set the lowest indices to prune the measurements
    uint16_t i;
    for (i = 0; i < numToPrune; i++) {
        unset_bit(LOWEST_IDX[i], collectedIndices);
    }
}
