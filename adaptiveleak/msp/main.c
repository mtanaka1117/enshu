#include "main.h"

#define MAX_NUM_SAMPLES 10
#define BUFFER_SIZE 5000

static FixedPoint DATA_BUFFER[SEQ_LENGTH * NUM_FEATURES];
static struct Vector featureVectors[SEQ_LENGTH];

static FixedPoint ZERO_BUFFER[NUM_FEATURES] = { 0 };
static struct Vector ZERO_FEATURES = { ZERO_BUFFER, NUM_FEATURES };


int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Must provide an input file.\n");
        return 0;
    }

    // Create the input buffer for file reading
    char buffer[BUFFER_SIZE];
    char *feature;
    FixedPoint featureVal;

    // Create the data buffer
    uint16_t i;
    for (i = 0; i < SEQ_LENGTH; i++) {
        featureVectors[i].data = DATA_BUFFER + (NUM_FEATURES * i);
        featureVectors[i].size = NUM_FEATURES;
    }

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

    // Indices to load data into feature vector array
    uint16_t sampleIdx;
    uint16_t featureIdx;

    uint32_t collectCount = 0;
    uint32_t totalCount = 0;
    uint32_t count = 0;
    uint32_t idx = 0;

    uint8_t shouldCollect = 0;

    // Open the input file
    FILE *fin = fopen(argv[1], "r");

    while (fgets(buffer, BUFFER_SIZE, fin) != NULL) {

        // Read all sequence elements from the file
        featureIdx = 0;
        sampleIdx = 0;

        feature = strtok(buffer, ",");
        while (feature != NULL) {
            featureVal = atoi(feature);

            featureVectors[sampleIdx].data[featureIdx] = featureVal;

            feature = strtok(NULL, ",");

            featureIdx++;

            if (featureIdx == NUM_FEATURES) {
                featureIdx = 0;
                sampleIdx++;
            }
        }

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
        for (i = 0; i < SEQ_LENGTH; i++) {
            #ifdef IS_UNIFORM
            shouldCollect = uniform_should_collect(&policy, i);
            #elif defined(IS_ADAPTIVE_HEURISTIC)
            shouldCollect = heuristic_should_collect(&policy, i);
            #elif defined(IS_ADAPTIVE_DEVIATION)
            shouldCollect = deviation_should_collect(&policy, i);
            #elif defined(IS_SKIP_RNN)
            shouldCollect = skip_rnn_should_collect(&policy, i);
            #endif

            if (shouldCollect) {
                collectCount++;
                count++;

                #ifdef IS_ADAPTIVE_HEURISTIC
                heuristic_update(&policy, featureVectors + i, prevFeatures);
                prevFeatures = featureVectors + i;
                #elif defined(IS_ADAPTIVE_DEVIATION)
                deviation_update(&policy, featureVectors + i, DEFAULT_PRECISION);
                #elif defined(IS_SKIP_RNN)
                skip_rnn_update(&policy, featureVectors + i, DEFAULT_PRECISION);
                #endif
            }

            totalCount++;
        }

        printf("%d ", count);
        idx++;
    }

    fclose(fin);
    printf("\n");

    float rate = ((float) collectCount) / ((float) totalCount);
    printf("Collection Rate: %d / %d (%f)\n", collectCount, totalCount, rate);

    return 0;
}
