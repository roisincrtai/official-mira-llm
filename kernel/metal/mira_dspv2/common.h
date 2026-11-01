#pragma once
#include <metal_stdlib>
using namespace metal;

#ifndef MIRA_ORDER
#define MIRA_ORDER 2
#endif
#ifndef MIRA_MEMORY
#define MIRA_MEMORY 16
#endif
#ifndef MIRA_VALUE
#define MIRA_VALUE float
#endif

#define MIRA_NAME_(name, order, memory, value) name##_k##order##_m##memory##_##value
#define MIRA_NAME(name, order, memory, value) MIRA_NAME_(name, order, memory, value)
#define MIRA_KERNEL(name) MIRA_NAME(name, MIRA_ORDER, MIRA_MEMORY, MIRA_VALUE)

#define ORDER MIRA_ORDER
#define MEMORY MIRA_MEMORY
#define SIMD 32
#define LANES (MEMORY < SIMD ? MEMORY : SIMD)
#define COLUMNS (MEMORY / LANES)
#define REDUCED (ORDER + 2)
#define SHARED_FLOATS 4096
#define MAX_SIMDGROUPS (SHARED_FLOATS / (REDUCED * MEMORY) < SIMD ? SHARED_FLOATS / (REDUCED * MEMORY) : SIMD)

#define P_BATCH 0
#define P_TOKENS 1
#define P_CHANNELS 2
#define P_TILES 3
#define P_GROUP 4
#define P_RADIUS 5
#define P_ROTATION 6
#define P_CHUNK 7
#define P_CHUNKS 8
#define P_FLAGS 9
#define P_ROW_X 10
#define P_ROW_GAMMA 11
#define P_ROW_ALPHA 12
#define P_ROW_BETA 13
#define P_ROW_RHO 14
#define P_ROW_WRITE 15
#define P_ROW_READ 16
#define P_COUNT 17

#define HAS_STATE 1
#define HAS_FINAL 2
#define HAS_VALID 4
#define SAVE_CHECKPOINT 8

typedef MIRA_VALUE value_t;

struct lattice_shape {
    int batch;
    int tokens;
    int channels;
    int tiles;
    int group;
    int radius;
    int rotation;
    int chunk;
    int chunks;
    int flags;
    ulong row_x;
    ulong row_gamma;
    ulong row_alpha;
    ulong row_beta;
    ulong row_rho;
    ulong row_write;
    ulong row_read;
};

static inline lattice_shape read_shape(constant int *params) {
    lattice_shape shape;
    shape.batch = params[P_BATCH];
    shape.tokens = params[P_TOKENS];
    shape.channels = params[P_CHANNELS];
    shape.tiles = params[P_TILES];
    shape.group = params[P_GROUP];
    shape.radius = params[P_RADIUS];
    shape.rotation = params[P_ROTATION];
    shape.chunk = params[P_CHUNK];
    shape.chunks = params[P_CHUNKS];
    shape.flags = params[P_FLAGS];
    shape.row_x = ulong(params[P_ROW_X]);
    shape.row_gamma = ulong(params[P_ROW_GAMMA]);
    shape.row_alpha = ulong(params[P_ROW_ALPHA]);
    shape.row_beta = ulong(params[P_ROW_BETA]);
    shape.row_rho = ulong(params[P_ROW_RHO]);
    shape.row_write = ulong(params[P_ROW_WRITE]);
    shape.row_read = ulong(params[P_ROW_READ]);
    return shape;
}

static inline float sigmoid_of(float value) {
    const float decay = precise::exp(-abs(value));
    const float inverse = 1.0f / (1.0f + decay);
    return value >= 0.0f ? inverse : decay * inverse;
}

static inline float silu_of(float value) {
    return value * sigmoid_of(value);
}

static inline float lane_sum(float value) {
    for (ushort offset = LANES / 2; offset > 0; offset >>= 1) {
        value += simd_shuffle_xor(value, offset);
    }
    return value;
}

static inline float slot_sum(float value) {
    for (ushort offset = LANES; offset < SIMD; offset <<= 1) {
        value += simd_shuffle_xor(value, offset);
    }
    return value;
}
