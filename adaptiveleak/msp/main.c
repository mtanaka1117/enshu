#include "main.h"

static FixedPoint DATA_BUFFER[SEQ_LENGTH * NUM_FEATURES];
static struct Vector featureVectors[SEQ_LENGTH];

static FixedPoint ZERO_BUFFER[NUM_FEATURES];
static struct Vector ZERO_FEATURES = { ZERO_BUFFER, NUM_FEATURES };

#ifdef IS_GROUP_ENCODED
static FixedPoint FEATURE_BUFFER[SEQ_LENGTH * NUM_FEATURES];
static int8_t SHIFT_BUFFER[SEQ_LENGTH * NUM_FEATURES];
static uint16_t COUNT_BUFFER[SEQ_LENGTH * NUM_FEATURES];
#endif


int main(void) {
    char *feature;
    FixedPoint featureVal;

    // Create the data buffer
    uint16_t i;
    for (i = 0; i < SEQ_LENGTH; i++) {
        featureVectors[i].data = DATA_BUFFER + (NUM_FEATURES * i);
        featureVectors[i].size = NUM_FEATURES;
    }

    for (i = 0; i < NUM_FEATURES; i++) {
        ZERO_BUFFER[i] = 0;
    }

    // Make the bitmap for this sequence
    uint16_t numBytes = (SEQ_LENGTH / BITS_PER_BYTE);
    if ((SEQ_LENGTH % BITS_PER_BYTE) > 0) {
        numBytes += 1;
    }

    uint8_t collectedBuffer[numBytes];
    struct BitMap collectedIndices = { collectedBuffer, numBytes };

    uint16_t numEncodedBytes = 0;
    uint8_t outputBuffer[1024];

    // Make the policy
    #ifdef IS_UNIFORM
    struct UniformPolicy policy;
    uniform_policy_init(&policy, COLLECT_INDICES, NUM_INDICES);
    #elif defined(IS_ADAPTIVE_HEURISTIC)
    struct Vector *prevFeatures;
    
    struct HeuristicPolicy policy;
    heuristic_policy_init(&policy, MAX_SKIP, THRESHOLD);
    #elif defined(IS_ADAPTIVE_DEVIATION)
    struct DeviationPolicy policy;

    FixedPoint meanData[NUM_FEATURES];
    struct Vector mean = { meanData, NUM_FEATURES };

    FixedPoint devData[NUM_FEATURES];
    struct Vector dev = { devData, NUM_FEATURES };

    deviation_policy_init(&policy, MAX_SKIP, THRESHOLD, ALPHA, BETA, &mean, &dev);
    #elif defined(IS_SKIP_RNN)
    struct SkipRNNPolicy policy;

    FixedPoint stateData[STATE_SIZE];
    struct Vector state = { stateData, STATE_SIZE };

    skip_rnn_policy_init(&policy, &W_CANDIDATE, &B_CANDIDATE, &W_UPDATE, &B_UPDATE, &W_STATE, B_STATE, &state, &INITIAL_STATE, &MEAN, &SCALE, RNN_PRECISION);
    #endif

    // Indices to sample data
    uint16_t seqIdx;
    uint16_t elemIdx;

    uint32_t collectCount = 0;
    uint32_t totalCount = 0;
    uint32_t count = 0;
    uint32_t idx = 0;

    uint8_t shouldCollect = 0;
    uint8_t didCollect = 1;

    for (seqIdx = 0; seqIdx < 3; seqIdx++) {
        // Clear the collected bit map
        clear_bitmap(&collectedIndices);

        #ifdef IS_UNIFORM
        uniform_reset(&policy);
        #elif defined(IS_ADAPTIVE_HEURISTIC)
        heuristic_reset(&policy);
        prevFeatures = &ZERO_FEATURES;
        #elif defined(IS_ADAPTIVE_DEVIATION)
        deviation_reset(&policy);
        #elif defined(IS_SKIP_RNN)
        skip_rnn_reset(&policy, &INITIAL_STATE);
        #endif

        count = 0;

        // Iterate through the elements and select elements to keep.
        for (elemIdx = 0; elemIdx < SEQ_LENGTH; elemIdx++) {
            #ifdef IS_UNIFORM
            shouldCollect = uniform_should_collect(&policy, elemIdx);
            #elif defined(IS_ADAPTIVE_HEURISTIC)
            shouldCollect = heuristic_should_collect(&policy, elemIdx);
            #elif defined(IS_ADAPTIVE_DEVIATION)
            shouldCollect = deviation_should_collect(&policy, elemIdx);
            #elif defined(IS_SKIP_RNN)
            shouldCollect = skip_rnn_should_collect(&policy, elemIdx);
            #endif

            if (shouldCollect) {
                collectCount++;
                count++;

                // Collect the data
                didCollect = get_measurement((featureVectors + elemIdx)->data, seqIdx, elemIdx, NUM_FEATURES, SEQ_LENGTH);

                if (!didCollect) {
                    printf("ERROR. Could not collect data at Seq %d, Element %d\n", seqIdx, elemIdx);
                    break;
                }

                // Record the collection of this element.
                set_bit(elemIdx, &collectedIndices);

                #ifdef IS_UNIFORM
                uniform_update(&policy);
                #elif defined(IS_ADAPTIVE_HEURISTIC)
                heuristic_update(&policy, featureVectors + elemIdx, prevFeatures);
                prevFeatures = featureVectors + elemIdx;
                #elif defined(IS_ADAPTIVE_DEVIATION)
                deviation_update(&policy, featureVectors + elemIdx, DEFAULT_PRECISION);
                #elif defined(IS_SKIP_RNN)
                skip_rnn_update(&policy, featureVectors + elemIdx, DEFAULT_PRECISION);
                #endif
            }

            totalCount++;
        }

        if (!didCollect) {
            break;
        }

        // Encode the collected elements.
        #ifdef IS_STANDARD_ENCODED
        numEncodedBytes = encode_standard(outputBuffer, featureVectors, &collectedIndices, NUM_FEATURES, SEQ_LENGTH);
        #elif defined(IS_GROUP_ENCODED)
        numEncodedBytes = encode_group(outputBuffer, featureVectors, &collectedIndices, count, NUM_FEATURES, SEQ_LENGTH, SIZE_BYTES, TARGET_BYTES, DEFAULT_PRECISION, MAX_COLLECTED, FEATURE_BUFFER, SHIFT_BUFFER, COUNT_BUFFER, 1);
        #endif

        print_message(outputBuffer, numEncodedBytes);

        //printf("%d ", count);
        printf("\n");
    }

    printf("\n");

    float rate = ((float) collectCount) / ((float) totalCount);
    printf("Collection Rate: %d / %d (%f)\n", collectCount, totalCount, rate);

    return 0;
}


void print_message(uint8_t *buffer, uint16_t numBytes) {
    uint16_t i;
    for (i = 0; i < numBytes; i++) {
        printf("\\x%02x", buffer[i]);
    }
    printf("\n");
    printf("Num Bytes: %d\n", numBytes);
}


