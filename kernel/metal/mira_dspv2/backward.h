#pragma once
#include "stage_maps.h"

kernel void MIRA_KERNEL(mira_backward)(
    device const value_t *x [[buffer(0)]],
    device const value_t *alpha [[buffer(1)]],
    device const value_t *beta [[buffer(2)]],
    device const value_t *rho [[buffer(3)]],
    device const value_t *raw_gamma [[buffer(4)]],
    device const value_t *memory_write [[buffer(5)]],
    device const value_t *memory_read [[buffer(6)]],
    device const uchar *valid [[buffer(7)]],
    device const float *checkpoint [[buffer(8)]],
    device const value_t *grad_output [[buffer(9)]],
    device float *scratch [[buffer(10)]],
    device value_t *grad_x [[buffer(11)]],
    device value_t *grad_alpha [[buffer(12)]],
    device value_t *grad_beta [[buffer(13)]],
    device value_t *grad_rho [[buffer(14)]],
    device value_t *grad_gamma [[buffer(15)]],
    device float *partial [[buffer(16)]],
    constant int *params [[buffer(17)]],
    uint group_index [[threadgroup_position_in_grid]],
    uint thread_index [[thread_position_in_threadgroup]]) {
    threadgroup float staging[MAX_SIMDGROUPS * REDUCED * MEMORY];

    const lattice_shape shape = read_shape(params);
    const int sequence = int(group_index) / shape.tiles;
    const int tile = int(group_index) % shape.tiles;
    const int lane = int(thread_index) % LANES;
    const int simd_index = int(thread_index) / SIMD;
    const bool leader = int(thread_index) % SIMD < LANES;
    const int simdgroups = shape.group / SIMD;
    const int channel = tile * (shape.group / LANES) + int(thread_index) / LANES;
    const bool active = channel < shape.channels;
    const int source = active ? channel : shape.channels - 1;
    const ulong channels = ulong(shape.channels);
    const ulong threads = ulong(shape.batch) * ulong(shape.tiles) * ulong(shape.group);
    const ulong worker = ulong(group_index) * ulong(shape.group) + ulong(thread_index);
    const bool masked = (shape.flags & HAS_VALID) != 0;

    float adjoint[COLUMNS][ORDER];
    for (int column = 0; column < COLUMNS; ++column) {
        for (int rung = 0; rung < ORDER; ++rung) {
            adjoint[column][rung] = 0.0f;
        }
    }

    for (int block = shape.chunks - 1; block >= 0; --block) {
        const int first = block * shape.chunk;
        const int last = min(first + shape.chunk, shape.tokens);
        const ulong base = ulong(sequence) * ulong(shape.chunks) + ulong(block);

        float state[COLUMNS][ORDER];
        for (int column = 0; column < COLUMNS; ++column) {
            const ulong memory = ulong(column * LANES + lane);
            for (int rung = 0; rung < ORDER; ++rung) {
                state[column][rung] =
                    checkpoint[((base * ORDER + rung) * channels + source) * MEMORY + memory];
            }
        }

        for (int token = first; token < last; ++token) {
            const ulong row = ulong(sequence) * ulong(shape.tokens) + ulong(token);
            for (int rung = 0; rung < ORDER; ++rung) {
                for (int column = 0; column < COLUMNS; ++column) {
                    scratch[((ulong(token - first) * ORDER + rung) * COLUMNS + column) * threads + worker] =
                        state[column][rung];
                }
            }
            if (masked && valid[row] == 0) {
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
            const float signal = float(x[row * shape.row_x + source]);
            for (int column = 0; column < COLUMNS; ++column) {
                const ulong memory = ulong(column * LANES + lane);
                float wave = signal * float(memory_write[row * shape.row_write + memory]);
                for (int index = ORDER - 1; index >= 0; --index) {
                    const float delayed = state[column][index];
                    const float rung = fma(scaled_cosine[index], delayed, scaled_sine[index] * wave);
                    wave = fma(scaled_cosine[index], wave, -(scaled_sine[index] * delayed));
                    if (index + 1 < ORDER) {
                        state[column][index + 1] = rung;
                    }
                }
                state[column][0] = wave;
            }
        }

        for (int token = last - 1; token >= first; --token) {
            const ulong row = ulong(sequence) * ulong(shape.tokens) + ulong(token);
            const ulong slot = (row * ulong(shape.tiles) + ulong(tile)) * (REDUCED * MEMORY);
            if (masked && valid[row] == 0) {
                if (active && lane == 0) {
                    const ulong cell = row * channels + ulong(channel);
                    grad_x[cell] = value_t(0.0f);
                    grad_gamma[cell] = value_t(0.0f);
                    for (int index = 0; index < ORDER; ++index) {
                        grad_alpha[cell * ORDER + index] = value_t(0.0f);
                        grad_beta[cell * ORDER + index] = value_t(0.0f);
                    }
                    for (int rung = 0; rung <= ORDER; ++rung) {
                        grad_rho[cell * (ORDER + 1) + rung] = value_t(0.0f);
                    }
                }
                for (int index = int(thread_index); index < REDUCED * MEMORY; index += shape.group) {
                    partial[slot + ulong(index)] = 0.0f;
                }
                continue;
            }

            stage_value stages[ORDER];
            float scaled_cosine[ORDER];
            float scaled_sine[ORDER];
            for (int index = 0; index < ORDER; ++index) {
                stages[index] = make_stage(
                    shape.radius, shape.rotation,
                    float(alpha[row * shape.row_alpha + ulong(source) * ORDER + index]),
                    float(beta[row * shape.row_beta + ulong(source) * ORDER + index]));
                scaled_cosine[index] = stages[index].radius * stages[index].cosine;
                scaled_sine[index] = stages[index].radius * stages[index].sine;
            }
            float weight[ORDER + 1];
            for (int rung = 0; rung <= ORDER; ++rung) {
                weight[rung] = float(rho[row * shape.row_rho + ulong(source) * (ORDER + 1) + rung]);
            }
            const float signal = float(x[row * shape.row_x + source]);
            const float gate_input = float(raw_gamma[row * shape.row_gamma + source]);
            const float gate_sigmoid = sigmoid_of(gate_input);
            const float upstream = active ? float(grad_output[row * channels + ulong(channel)]) : 0.0f;
            const float gated = upstream * gate_input * gate_sigmoid;
            float sensitivity[ORDER + 1];
            for (int rung = 0; rung <= ORDER; ++rung) {
                sensitivity[rung] = gated * weight[rung];
            }

            float grad_scaled_cosine[ORDER];
            float grad_scaled_sine[ORDER];
            for (int index = 0; index < ORDER; ++index) {
                grad_scaled_cosine[index] = 0.0f;
                grad_scaled_sine[index] = 0.0f;
            }
            float readout[ORDER + 1];
            for (int rung = 0; rung <= ORDER; ++rung) {
                readout[rung] = 0.0f;
            }
            float grad_signal = 0.0f;
            float contribution[REDUCED][COLUMNS];

            for (int column = 0; column < COLUMNS; ++column) {
                const ulong memory = ulong(column * LANES + lane);
                const float write_value = float(memory_write[row * shape.row_write + memory]);
                float read_value[ORDER + 1];
                for (int rung = 0; rung <= ORDER; ++rung) {
                    read_value[rung] = float(memory_read[row * shape.row_read + ulong(rung) * MEMORY + memory]);
                }
                float delayed[ORDER];
                for (int index = 0; index < ORDER; ++index) {
                    delayed[index] =
                        scratch[((ulong(token - first) * ORDER + index) * COLUMNS + column) * threads + worker];
                }
                float incoming[ORDER];
                float rungs[ORDER + 1];
                float wave = signal * write_value;
                for (int index = ORDER - 1; index >= 0; --index) {
                    incoming[index] = wave;
                    rungs[index + 1] = fma(scaled_cosine[index], delayed[index], scaled_sine[index] * wave);
                    wave = fma(scaled_cosine[index], wave, -(scaled_sine[index] * delayed[index]));
                }
                rungs[0] = wave;
                for (int rung = 0; rung <= ORDER; ++rung) {
                    readout[rung] = fma(read_value[rung], rungs[rung], readout[rung]);
                    contribution[rung][column] = active ? sensitivity[rung] * rungs[rung] : 0.0f;
                }
                float grad_wave = fma(sensitivity[0], read_value[0], adjoint[column][0]);
                for (int index = 0; index < ORDER; ++index) {
                    float grad_level = sensitivity[index + 1] * read_value[index + 1];
                    if (index + 1 < ORDER) {
                        grad_level += adjoint[column][index + 1];
                    }
                    const float previous = delayed[index];
                    const float upper = incoming[index];
                    grad_scaled_cosine[index] += fma(grad_level, previous, grad_wave * upper);
                    grad_scaled_sine[index] += fma(grad_level, upper, -(grad_wave * previous));
                    adjoint[column][index] =
                        fma(scaled_cosine[index], grad_level, -(scaled_sine[index] * grad_wave));
                    grad_wave = fma(scaled_sine[index], grad_level, scaled_cosine[index] * grad_wave);
                }
                grad_signal = fma(grad_wave, write_value, grad_signal);
                contribution[ORDER + 1][column] = active ? grad_wave * signal : 0.0f;
            }

            for (int index = 0; index < ORDER; ++index) {
                grad_scaled_cosine[index] = lane_sum(grad_scaled_cosine[index]);
                grad_scaled_sine[index] = lane_sum(grad_scaled_sine[index]);
            }
            for (int rung = 0; rung <= ORDER; ++rung) {
                readout[rung] = lane_sum(readout[rung]);
            }
            grad_signal = lane_sum(grad_signal);

            if (active && lane == 0) {
                const ulong cell = row * channels + ulong(channel);
                float total = 0.0f;
                for (int rung = 0; rung <= ORDER; ++rung) {
                    total = fma(weight[rung], readout[rung], total);
                    grad_rho[cell * (ORDER + 1) + rung] = value_t(gated * readout[rung]);
                }
                grad_gamma[cell] = value_t(
                    upstream * total * gate_sigmoid * (1.0f + gate_input * (1.0f - gate_sigmoid)));
                grad_x[cell] = value_t(grad_signal);
                for (int index = 0; index < ORDER; ++index) {
                    const stage_value value = stages[index];
                    const float grad_radius =
                        grad_scaled_cosine[index] * value.cosine + grad_scaled_sine[index] * value.sine;
                    grad_alpha[cell * ORDER + index] = value_t(alpha_gradient(
                        shape.rotation, value, grad_scaled_cosine[index] * value.radius,
                        grad_scaled_sine[index] * value.radius));
                    grad_beta[cell * ORDER + index] = value_t(grad_radius * value.slope);
                }
            }

            for (int entry = 0; entry < REDUCED; ++entry) {
                for (int column = 0; column < COLUMNS; ++column) {
                    const float value = slot_sum(contribution[entry][column]);
                    if (leader) {
                        staging[(simd_index * REDUCED + entry) * MEMORY + column * LANES + lane] = value;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int index = int(thread_index); index < REDUCED * MEMORY; index += shape.group) {
                float total = 0.0f;
                for (int part = 0; part < simdgroups; ++part) {
                    total += staging[part * REDUCED * MEMORY + index];
                }
                partial[slot + ulong(index)] = total;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
