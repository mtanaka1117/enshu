#include "sampler.h"


uint8_t get_measurement(FixedPoint *result, uint16_t seqNum, uint16_t elemNum, uint16_t numFeatures, uint16_t seqLength) {
    uint32_t featuresPerSeq = ((uint32_t) numFeatures) * ((uint32_t) seqLength);
    uint16_t dataIdx = seqNum * featuresPerSeq + elemNum * numFeatures;

    if ((dataIdx + numFeatures) >= DATASET_LENGTH) {
        return 0;
    }

    uint16_t i, j;
    for (i = numFeatures; i > 0; i--) {
        j = i - 1;

        #ifdef IS_MSP
        result[j] = __data20_read_short((unsigned long int) (DATASET + dataIdx + j));
        #else
        result[j] = DATASET[dataIdx + j];
        #endif
    }

    return 1;
}
