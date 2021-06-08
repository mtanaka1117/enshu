#include <stdint.h>

#include "utils/fixed_point.h"
#include "utils/matrix.h"

#ifndef POLICY_H_
#define POLICY_H_

struct UniformPolicy {
    uint16_t collectIdx;
    const uint16_t *collectIndices;
    uint16_t numIndices;
};

struct HeuristicPolicy {
    uint16_t maxSkip;
    uint16_t currentSkip;
    uint16_t sampleSkip;
    FixedPoint threshold;
};

struct DeviationPolicy {
    uint16_t maxSkip;
    uint16_t currentSkip;
    uint16_t sampleSkip;
    FixedPoint threshold;
    FixedPoint alpha;
    FixedPoint beta;
    struct Vector *mean;
    struct Vector *dev;
};

// Uniform Policy Operations
void uniform_policy_init(struct UniformPolicy *policy, const uint16_t *collectIndices, uint16_t numIndices);
uint8_t uniform_should_collect(struct UniformPolicy *policy, uint16_t seqIdx);
void uniform_reset(struct UniformPolicy *policy);

// Heuristic Policy Operations
void heuristic_policy_init(struct HeuristicPolicy *policy, uint16_t maxSkip, FixedPoint threshold);
uint8_t heuristic_should_collect(struct HeuristicPolicy *policy, uint16_t seqIdx);
void heuristic_update(struct HeuristicPolicy *policy, struct Vector *curr, struct Vector *prev);
void heuristic_reset(struct HeuristicPolicy *policy);


// Deviation Policy Operations
void deviation_policy_init(struct DeviationPolicy *policy, uint16_t maxSkip, FixedPoint threshold, FixedPoint alpha, FixedPoint beta, struct Vector *mean, struct Vector *dev);
uint8_t deviation_should_collect(struct DeviationPolicy *policy, uint16_t seqIdx);
void deviation_update(struct DeviationPolicy *policy, struct Vector *curr, uint16_t precision);
void deviation_reset(struct DeviationPolicy *policy);

#endif
