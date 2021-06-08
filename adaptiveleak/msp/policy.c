#include "policy.h"

static FixedPoint POLICY_BUFFER[20];


/**
 * Uniform Policy Functions
 */

uint8_t uniform_should_collect(struct UniformPolicy *policy, uint16_t seqIdx) {
    if (policy->collectIdx >= policy->numIndices) {
        return 0;
    }

    uint8_t result = (seqIdx == policy->collectIndices[policy->collectIdx]);
    policy->collectIdx += result;
    return result;
}


void uniform_reset(struct UniformPolicy *policy) {
    policy->collectIdx = 0;
}

/**
 * Heuristic Policy Functions
 */

uint8_t heuristic_should_collect(struct HeuristicPolicy *policy uint16_t seqIdx) {
    uint8_t result = (policy->sampleSkip == 0);
    policy->sampleSkip -= 1;
    return result;
}


void heuristic_update(struct HeuristicPolicy *policy, struct Vector *curr, struct Vector *prev) {
    FixedPoint norm = vector_diff_norm(curr, prev);

    if (norm >= policy->threshold) {
        policy->currentSkip = 0;
    } else {
        uint16_t nextSkip = policy->currentSkip + 1;
        uint8_t cond = nextSkip < policy->maxSkip;
        policy->currentSkip = cond * nextSkip + (1 - cond) * policy->maxSkip;
    }

    policy->sampleSkip = policy->currentSkip;
}


void heuristic_reset(struct HeuristicPolicy *policy)  {
    policy->sampleSkip = 0;
    policy->currentSkip = 0;
}


/**
 * Deviation Policy Functions
 */
uint8_t deviation_should_collect(struct DeviationPolicy *policy uint16_t seqIdx) {
    uint8_t result = (policy->sampleSkip == 0);
    policy->sampleSkip -= 1;
    return result;
}


void deviation_update(struct DeviationPolicy *policy, struct Vector *curr, struct Vector *prev, uint16_t precision) {
    policy->mean = vector_gated_add(policy->mean, curr, policy->mean, policy->alpha, precision);

    struct Vector temp = { POLICY_BUFFER, curr->size };
    vector_absolute_diff(&temp, &temp, policy->mean);

    policy->dev = vector_gated_add(policy->dev, &temp, policy->dev, policy->beta, precision);

    FixedPoint norm = vector_norm(policy->dev);

    if (norm >= policy->threshold) {
        policy->currentSkip = (policy->currentSkip >> 1);
    } else {
        uint16_t nextSkip = policy->currentSkip + 1;
        uint8_t cond = nextSkip < policy->maxSkip;
        policy->currentSkip = cond * nextSkip + (1 - cond) * policy->maxSkip;
    }

    policy->sampleSkip = policy->currentSkip;
}


void deviation_reset(struct DeviationPolicy *policy)  {
    policy->sampleSkip = 0;
    policy->currentSkip = 0;
    vector_set(policy->mean, 0);
    vector_set(policy->dev, 0);
}
