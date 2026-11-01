#pragma once
#include "common.h"

#define RADIUS_CAUCHY 0
#define RADIUS_GAUSSIAN 1
#define RADIUS_EXPEXP 2
#define RADIUS_EXPSOFTPLUS 3
#define RADIUS_HILL 4

#define ROTATION_MOBIUS 0
#define ROTATION_HYPERBOLIC 1
#define ROTATION_IDENTITY 2

struct stage_value {
    float radius;
    float cosine;
    float sine;
    float slope;
    float rate;
};

static inline float stage_radius(int kind, float beta, thread float &slope) {
    if (kind == RADIUS_GAUSSIAN) {
        const float radius = precise::exp(-beta * beta);
        slope = -2.0f * beta * radius;
        return radius;
    }
    if (kind == RADIUS_EXPEXP) {
        const float growth = precise::exp(min(beta, 80.0f));
        const float radius = precise::exp(-growth);
        slope = beta < 80.0f ? -growth * radius : 0.0f;
        return radius;
    }
    if (kind == RADIUS_EXPSOFTPLUS) {
        const float radius = sigmoid_of(-beta);
        slope = -radius * (1.0f - radius);
        return radius;
    }
    if (kind == RADIUS_HILL) {
        const float square = beta * beta;
        const float complement = 1.0f / (1.0f + square);
        slope = 2.0f * beta * complement * complement;
        return square > 1.0f ? 1.0f - complement : square * complement;
    }
    const float radius = 1.0f / fma(beta, beta, 1.0f);
    slope = -2.0f * beta * radius * radius;
    return radius;
}

static inline void stage_rotation(int kind, float alpha, thread float &cosine, thread float &sine,
                           thread float &rate) {
    if (kind == ROTATION_HYPERBOLIC) {
        const float decay = precise::exp(-abs(alpha));
        const float decay_squared = decay * decay;
        const float inverse = 1.0f / (1.0f + decay_squared);
        cosine = 2.0f * decay * inverse;
        sine = copysign((1.0f - decay_squared) * inverse, alpha);
        rate = 0.0f;
        return;
    }
    if (kind == ROTATION_IDENTITY) {
        cosine = precise::cos(alpha);
        sine = precise::sin(alpha);
        rate = 0.0f;
        return;
    }
    const bool near_zero = abs(alpha) <= 1.0f;
    const float base = near_zero ? alpha : 1.0f / alpha;
    const float square = base * base;
    const float inverse = 1.0f / (1.0f + square);
    cosine = 2.0f * base * inverse;
    sine = (near_zero ? 1.0f - square : square - 1.0f) * inverse;
    rate = 2.0f * (near_zero ? inverse : square * inverse);
}

static inline stage_value make_stage(int radius_kind, int rotation_kind, float alpha, float beta) {
    stage_value value;
    value.radius = stage_radius(radius_kind, beta, value.slope);
    stage_rotation(rotation_kind, alpha, value.cosine, value.sine, value.rate);
    return value;
}

static inline float alpha_gradient(int rotation_kind, stage_value value, float grad_cosine, float grad_sine) {
    if (rotation_kind == ROTATION_HYPERBOLIC) {
        return value.cosine * (grad_sine * value.cosine - grad_cosine * value.sine);
    }
    if (rotation_kind == ROTATION_IDENTITY) {
        return grad_sine * value.cosine - grad_cosine * value.sine;
    }
    return (grad_cosine * value.sine - grad_sine * value.cosine) * value.rate;
}
