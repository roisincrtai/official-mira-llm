#pragma once
#include "stage_maps.h"

kernel void MIRA_KERNEL(mira_forward)(
    device const value_t *x [[buffer(0)]],
    device const value_t *alpha [[buffer(1)]],
    device const value_t *beta [[buffer(2)]],
    device const value_t *rho [[buffer(3)]],
    device const value_t *raw_gamma [[buffer(4)]],
    device const value_t *memory_write [[buffer(5)]],
    device const value_t *memory_read [[buffer(6)]],
    device const uchar *valid [[buffer(7)]],
    device const float *initial [[buffer(8)]],
    device value_t *output [[buffer(9)]],
    device float *final_state [[buffer(10)]],
    device float *checkpoint [[buffer(11)]],
    constant int *params [[buffer(12)]],
    uint group_index [[threadgroup_position_in_grid]],
    uint thread_index [[thread_position_in_threadgroup]]) {
    const lattice_shape shape = read_shape(params);
    const int sequence = int(group_index) / shape.tiles;
    const int tile = int(group_index) % shape.tiles;
    const int lane = int(thread_index) % LANES;
    const int channel = tile * (shape.group / LANES) + int(thread_index) / LANES;
    const bool active = channel < shape.channels;
    const int source = active ? channel : shape.channels - 1;
    const ulong channels = ulong(shape.channels);

    float state[COLUMNS][ORDER];
    for (int column = 0; column < COLUMNS; ++column) {
        const ulong memory = ulong(column * LANES + lane);
        for (int rung = 0; rung < ORDER; ++rung) {
            state[column][rung] = (shape.flags & HAS_STATE)
                ? initial[((ulong(sequence) * ORDER + rung) * channels + source) * MEMORY + memory]
                : 0.0f;
        }
    }

    for (int token = 0; token < shape.tokens; ++token) {
        const ulong row = ulong(sequence) * ulong(shape.tokens) + ulong(token);
        if ((shape.flags & SAVE_CHECKPOINT) && token % shape.chunk == 0 && active) {
            const ulong base = ulong(sequence) * ulong(shape.chunks) + ulong(token / shape.chunk);
            for (int column = 0; column < COLUMNS; ++column) {
                const ulong memory = ulong(column * LANES + lane);
                for (int rung = 0; rung < ORDER; ++rung) {
                    checkpoint[((base * ORDER + rung) * channels + channel) * MEMORY + memory] =
                        state[column][rung];
                }
            }
        }
        const bool keep = !(shape.flags & HAS_VALID) || valid[row] != 0;
        if (!keep) {
            if (active && lane == 0) {
                output[row * channels + channel] = value_t(0.0f);
            }
            continue;
        }

        float scaled_cosine[ORDER];
        float scaled_sine[ORDER];
        for (int index = 0; index < ORDER; ++index) {
            const stage_value value = make_stage(
                shape.radius, shape.rotation,
                float(alpha[row * shape.row_alpha + ulong(source) * ORDER + index]),
                float(beta[row * shape.row_beta + ulong(source) * ORDER + index]));
            scaled_cosine[index] = value.radius * value.cosine;
            scaled_sine[index] = value.radius * value.sine;
        }
        float weight[ORDER + 1];
        for (int rung = 0; rung <= ORDER; ++rung) {
            weight[rung] = float(rho[row * shape.row_rho + ulong(source) * (ORDER + 1) + rung]);
        }
        const float signal = float(x[row * shape.row_x + source]);

        float total = 0.0f;
        for (int column = 0; column < COLUMNS; ++column) {
            const ulong memory = ulong(column * LANES + lane);
            float wave = signal * float(memory_write[row * shape.row_write + memory]);
            for (int index = ORDER - 1; index >= 0; --index) {
                const float delayed = state[column][index];
                const float rung = fma(scaled_cosine[index], delayed, scaled_sine[index] * wave);
                wave = fma(scaled_cosine[index], wave, -(scaled_sine[index] * delayed));
                total = fma(weight[index + 1] *
                            float(memory_read[row * shape.row_read + ulong(index + 1) * MEMORY + memory]),
                            rung, total);
                if (index + 1 < ORDER) {
                    state[column][index + 1] = rung;
                }
            }
            state[column][0] = wave;
            total = fma(weight[0] * float(memory_read[row * shape.row_read + memory]), wave, total);
        }
        total = lane_sum(total);
        if (active && lane == 0) {
            const float gate = silu_of(float(raw_gamma[row * shape.row_gamma + source]));
            output[row * channels + channel] = value_t(total * gate);
        }
    }

    if ((shape.flags & HAS_FINAL) && active) {
        for (int column = 0; column < COLUMNS; ++column) {
            const ulong memory = ulong(column * LANES + lane);
            for (int rung = 0; rung < ORDER; ++rung) {
                final_state[((ulong(sequence) * ORDER + rung) * channels + channel) * MEMORY + memory] =
                    state[column][rung];
            }
        }
    }
}
