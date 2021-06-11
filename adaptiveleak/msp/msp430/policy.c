#include "policy.h"

static FixedPoint POLICY_BUFFER[256];


/**
 * Uniform Policy Functions
 */
void uniform_policy_init(struct UniformPolicy *policy, const uint16_t *collectIndices, uint16_t numIndices) {
    policy->collectIndices = collectIndices;
    policy->numIndices = numIndices;
    policy->collectIdx = 0;
}

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
void heuristic_policy_init(struct HeuristicPolicy *policy, uint16_t maxSkip, FixedPoint threshold) {
    policy->maxSkip = maxSkip;
    policy->threshold = threshold;
    policy->currentSkip = 0;
    policy->sampleSkip = 0;
}

uint8_t heuristic_should_collect(struct HeuristicPolicy *policy, uint16_t seqIdx) {
    uint8_t result = (policy->sampleSkip == 0) || (seqIdx == 0);
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
void deviation_policy_init(struct DeviationPolicy *policy, uint16_t maxSkip, FixedPoint threshold, FixedPoint alpha, FixedPoint beta, struct Vector *mean, struct Vector *dev) {
    policy->maxSkip = maxSkip;
    policy->currentSkip = 0;
    policy->sampleSkip = 0;
    policy->threshold = threshold;
    policy->alpha = alpha;
    policy->beta = beta;
    policy->mean = mean;
    policy->dev = dev;
}


uint8_t deviation_should_collect(struct DeviationPolicy *policy, uint16_t seqIdx) {
    uint8_t result = (policy->sampleSkip == 0) || (seqIdx == 0);
    policy->sampleSkip -= 1;
    return result;
}


void deviation_update(struct DeviationPolicy *policy, struct Vector *curr, uint16_t precision) {
    policy->mean = vector_gated_add_scalar(policy->mean, curr, policy->mean, policy->alpha, precision);

    struct Vector temp = { POLICY_BUFFER, curr->size };
    vector_absolute_diff(&temp, curr, policy->mean);

    policy->dev = vector_gated_add_scalar(policy->dev, &temp, policy->dev, policy->beta, precision);
    FixedPoint norm = vector_norm(policy->dev);

    if (norm >= policy->threshold) {
        policy->currentSkip = (policy->currentSkip) >> 1;
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


/**
 * Skip RNN Policy Functions
 */
void skip_rnn_policy_init(struct SkipRNNPolicy *policy, struct Matrix *wCandidate, struct Vector *bCandidate, struct Matrix *wUpdate, struct Vector *bUpdate, struct Vector *wState, FixedPoint bState, struct Vector *state, struct Vector *initialState, struct Vector *mean, struct Vector *scale, uint16_t precision) {
    policy->updateProb = 0;
    policy->cumUpdateProb = 1 << precision;
    policy->threshold = 1 << (precision - 1);
    policy->wCandidate = wCandidate;
    policy->bCandidate = bCandidate;
    policy->wUpdate = wUpdate;
    policy->bUpdate = bUpdate;
    policy->wState = wState;
    policy->bState = bState;
    policy->mean = mean;
    policy->scale = scale;
    policy->precision = precision;

    policy->state = state;
    vector_copy(policy->state, initialState);
}


uint8_t skip_rnn_should_collect(struct SkipRNNPolicy *policy, uint16_t seqIdx) {
    uint8_t result = (policy->cumUpdateProb >= policy->threshold) || (seqIdx == 0);
    policy->cumUpdateProb = fp_add(policy->cumUpdateProb, policy->updateProb);
    return result;
}


void skip_rnn_update(struct SkipRNNPolicy *policy, struct Vector *curr, uint16_t featurePrecision) {
    // Normalize the features
    uint16_t numFeatures = curr->size;
    uint16_t rnnPrecision = policy->precision;

    uint16_t bufferOffset = 0;
    struct Vector normalized = { POLICY_BUFFER, numFeatures };
    bufferOffset += curr->size;

    vector_scale(&normalized, curr, policy->mean, policy->scale, featurePrecision, rnnPrecision);

    // Stack the state and input
    uint16_t stateSize = policy->state->size;
    struct Vector stacked = { POLICY_BUFFER + bufferOffset, numFeatures + stateSize };
    bufferOffset += numFeatures + stateSize;

    vector_stack(&stacked, &normalized, policy->state);

    // Compute the UGRNN transformations
    struct Vector candidate = { POLICY_BUFFER + bufferOffset, stateSize };
    bufferOffset += stateSize;

    matrix_vector_prod(&candidate, policy->wCandidate, &stacked, rnnPrecision);
    vector_add(&candidate, &candidate, policy->bCandidate);
    vector_apply(&candidate, &candidate, &fp_tanh, rnnPrecision);

    struct Vector updateGate = { POLICY_BUFFER + bufferOffset, stateSize };

    matrix_vector_prod(&updateGate, policy->wUpdate, &stacked, rnnPrecision);
    vector_add(&updateGate, &updateGate, policy->bUpdate);
    vector_apply(&updateGate, &updateGate, &fp_sigmoid, rnnPrecision);

    // Compute the next state
    vector_gated_add(policy->state, policy->state, &candidate, &updateGate, rnnPrecision);

    // Compute the update probability
    FixedPoint nextUpdate = vector_dot_prod(policy->state, policy->wState, rnnPrecision);
    nextUpdate = fp_add(nextUpdate, policy->bState);
    nextUpdate = fp_sigmoid(nextUpdate, rnnPrecision);

    policy->updateProb = nextUpdate;
    policy->cumUpdateProb = policy->updateProb;
}


void skip_rnn_reset(struct SkipRNNPolicy *policy, struct Vector *initialState) {
    policy->updateProb = 0;
    policy->cumUpdateProb = 1 << (policy->precision);
    vector_copy(policy->state, initialState);
}
