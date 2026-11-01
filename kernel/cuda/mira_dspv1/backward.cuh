#pragma once
#include "lattice.cuh"

template <typename IO, int N, int M, int L, int G, int MODE>
__device__ __forceinline__ void backward_phase_one(
    const IO *__restrict__ raw_alpha, const IO *__restrict__ raw_beta,
    const IO *__restrict__ raw_rho,
    const IO *__restrict__ gamma, const IO *__restrict__ memory_read,
    const IO *__restrict__ grad_output,
    const float *__restrict__ readout_sum,
    const bool *__restrict__ valid, float *__restrict__ reverse_carry,
    int batch, int tokens, int channels, int chunk,
    int row_stride, int alpha_row_stride, int beta_row_stride,
    int rho_row_stride, int gamma_row_stride, int read_row_stride, int stage_map) {
    const int chunks = (tokens + chunk - 1) / chunk;
    const int groups = (channels + G - 1) / G;
    const long long group_plane = blockIdx.x;
    const long long total = static_cast<long long>(batch) * chunks * groups;
    if (group_plane >= total) return;
    const int group = static_cast<int>(group_plane % groups);
    const int chunk_index = static_cast<int>((group_plane / groups) % chunks);
    const int batch_index = static_cast<int>(
        group_plane / (static_cast<long long>(groups) * chunks));
    const int lane = threadIdx.x % L;
    const int channel = group * G + threadIdx.x / L;
    const bool active = channel < channels;
    const long long plane = (static_cast<long long>(batch_index) * chunks + chunk_index)
                          * channels + channel;
    const int first = chunk_index * chunk;
    const int last = first + chunk < tokens ? first + chunk : tokens;
    float future[M / L][N] = {0.0f};
    extern __shared__ __align__(16) unsigned char scan_shared[];
    IO *read_cache = reinterpret_cast<IO *>(scan_shared);
    for (int tile_last = last; tile_last > first; tile_last -= ALPHA_TILE) {
        const int tile_first =
            (tile_last - ALPHA_TILE > first ? tile_last - ALPHA_TILE : first);
        const int tile_tokens = tile_last - tile_first;
        for (int index = threadIdx.x; index < tile_tokens * (N + 1) * M;
             index += blockDim.x) {
            const int local_token = index / ((N + 1) * M);
            const int rest = index % ((N + 1) * M);
            const int stage = rest / M;
            const int memory = rest % M;
            const long long row = static_cast<long long>(batch_index) * tokens
                                + tile_first + local_token;
            read_cache[index] =
                memory_read[row * read_row_stride + stage * M + memory];
        }
        __syncthreads();
        for (int token = tile_last - 1; token >= tile_first; --token) {
            const long long row = static_cast<long long>(batch_index) * tokens + token;
            if (valid && !valid[row]) continue;
            const long long readout_offset =
                row * rho_row_stride + channel * (N + 1);
            float slot[2 * N];
            float rho[N + 1];
            load_stage<IO, N, L>(
                raw_alpha, raw_beta, row, channel,
                alpha_row_stride, beta_row_stride, active, slot, stage_map);
            load_rho<IO, N, L>(raw_rho, readout_offset, active, rho);
            if (active) {
                /* grad_output and gamma are one scalar per channel-token and every lane needs
                 * the product, so each lane forms it from the same two broadcast loads rather
                 * than lane 0 forming it and broadcasting with __shfl_sync. */
                const float common =
                    load_value(grad_output + row * channels + channel)
                    * silu_fast(load_value(
                        gamma + row * gamma_row_stride + channel));
                float grad_u[M / L];
#pragma unroll
                for (int vector = 0; vector < M / L; ++vector) {
                    const int memory = lane + vector * L;
                    grad_u[vector] = fmaf(
                        common * rho[0],
                        load_value(read_cache + (token - tile_first) * (N + 1) * M
                                   + memory),
                        future[vector][0]);
                }
#pragma unroll
                for (int stage = 0; stage < N; ++stage) {
                    const float scaled_cosine = slot[stage];
                    const float scaled_sine = slot[N + stage];
#pragma unroll
                    for (int vector = 0; vector < M / L; ++vector) {
                        const int memory = lane + vector * L;
                        float grad_v = common * rho[stage + 1] * load_value(
                            read_cache + ((token - tile_first) * (N + 1) + stage + 1)
                            * M + memory);
                        if (stage + 1 < N)
                            grad_v += future[vector][stage + 1];
                        const float next = fmaf(
                            scaled_cosine, grad_v, -(scaled_sine * grad_u[vector]));
                        grad_u[vector] = fmaf(
                            scaled_cosine, grad_u[vector], scaled_sine * grad_v);
                        future[vector][stage] = next;
                    }
                }
            }
        }
        __syncthreads();
    }
    if (active) {
#pragma unroll
        for (int vector = 0; vector < M / L; ++vector) {
            const int memory = lane + vector * L;
            store_state<N>(reverse_carry + (plane * M + memory) * N, future[vector]);
        }
    }
}

template <typename IO, int N, int M, int L, int G, int S, int MODE>
__device__ __forceinline__ void backward_phase_three(
    const IO *__restrict__ x, const IO *__restrict__ raw_alpha,
    const IO *__restrict__ raw_beta,
    const IO *__restrict__ raw_rho, const IO *__restrict__ gamma,
    const IO *__restrict__ memory_write, const IO *__restrict__ memory_read,
    const float *__restrict__ checkpoint,
    const float *__restrict__ boundary, const IO *__restrict__ grad_output,
    IO *__restrict__ grad_x, IO *__restrict__ grad_alpha,
    IO *__restrict__ grad_beta,
    IO *__restrict__ grad_rho, IO *__restrict__ grad_gamma,
    float *__restrict__ grad_memory_groups, const bool *valid,
    int batch, int tokens, int channels, int chunk,
    int groups, int channel_group, int row_stride, int alpha_row_stride,
    int beta_row_stride, int rho_row_stride, int gamma_row_stride,
    int read_row_stride, int head_channels, int packed, int stage_map) {
    const int chunks = (tokens + chunk - 1) / chunk;
    const long long tile_plane = blockIdx.x;
    const long long total = static_cast<long long>(batch) * chunks * groups;
    if (tile_plane >= total) return;
    const int group = static_cast<int>(tile_plane % groups);
    const int chunk_index = static_cast<int>((tile_plane / groups) % chunks);
    const int batch_index = static_cast<int>(
        tile_plane / (static_cast<long long>(groups) * chunks));
    const int channel_first = group * head_channels;
    const int first = chunk_index * chunk;
    const int last = first + chunk < tokens ? first + chunk : tokens;
    const int lane = threadIdx.x % L;
    const int local_channel = threadIdx.x / L;
    const int segment_capacity = (chunk + S - 1) / S;
    const int segment_count = (last - first + S - 1) / S;
    const int span = MODE == 3 ? 4 * N : (MODE ? 2 * N : N);
    const long long grouped_values = static_cast<long long>(batch) * tokens
                                   * (N + 2) * M;
    extern __shared__ float shared[];
    float *segment_checkpoint = shared;
    /* THE SEGMENT'S REPLAYED STATE IS THREAD-PRIVATE AND LIVES IN REGISTERS, NOT SHARED. Each
     * thread wrote stored_state[((segment_token * G + local_channel) * N + stage) * M + memory]
     * and read back exactly the same element: same local_channel, same lane, same vector. It was
     * S * G * M * N floats of shared memory carrying data no other thread ever touched -- 32,768 B
     * of the 71,680 B that kept the G = 16 variant of this kernel unreachable at K = 4, d_m = 16.
     * In registers it is S * (M / L) * N per thread, 32 at the production shape against the 256 a
     * thread may hold at 256 threads and one block per SM. */
    float *stored_readout = segment_checkpoint + (segment_capacity - 1) * G * M * N;
    float *write_accumulator = stored_readout + S * G * (N + 1);
    float *read_accumulator = write_accumulator + chunk * M;
    float *write_cache = read_accumulator + chunk * (N + 1) * M;
    float *gradient_scratch = write_cache + chunk * M;
    float *alpha_cache = gradient_scratch + G * M * (N + 2)
        * (N == 2 && L == 16 ? 2 : 1);
    IO *read_cache = reinterpret_cast<IO *>(
        alpha_cache + chunk * G * span);
    const bool cache_read = N == 4 && sizeof(IO) == 2;
    for (int index = threadIdx.x; index < chunk * M; index += blockDim.x) {
        write_accumulator[index] = 0.0f;
        const int local_token = index / M;
        if (first + local_token < last) {
            const long long row = static_cast<long long>(batch_index) * tokens
                                + first + local_token;
            write_cache[index] = load_value(
                memory_write + row * M + index % M);
        } else {
            write_cache[index] = 0.0f;
        }
    }
    for (int index = threadIdx.x; index < chunk * (N + 1) * M; index += blockDim.x)
        read_accumulator[index] = 0.0f;
    if (cache_read) {
        for (int index = threadIdx.x; index < (last - first) * (N + 1) * M;
             index += blockDim.x) {
            const int local_token = index / ((N + 1) * M);
            const int rest = index % ((N + 1) * M);
            const int stage = rest / M;
            const int memory = rest % M;
            const long long row = static_cast<long long>(batch_index) * tokens
                                + first + local_token;
            read_cache[index] = memory_read[
                row * read_row_stride + stage * M + memory];
        }
    }
    __syncthreads();
    for (int channel_offset = 0; channel_offset < head_channels; channel_offset += G) {
        stage_alpha<IO, N, G, MODE, true>(raw_alpha, raw_beta, alpha_cache, batch_index, tokens, channels,
                              channel_first + channel_offset,
                              first, last - first,
                              alpha_row_stride, beta_row_stride, stage_map);
        __syncthreads();
        const int channel = channel_first + channel_offset + local_channel;
        const bool active = local_channel < G
                          && channel_offset + local_channel < head_channels
                          && channel < channels;
        const long long plane = (static_cast<long long>(batch_index) * chunks + chunk_index)
                              * channels + channel;
        float state[M / L][N] = {0.0f};
        float initial[M / L][N] = {0.0f};
        if (active) {
#pragma unroll
            for (int vector = 0; vector < M / L; ++vector) {
                const int memory = lane + vector * L;
                load_state<N>(checkpoint + (plane * M + memory) * N, initial[vector]);
#pragma unroll
                for (int stage = 0; stage < N; ++stage)
                    state[vector][stage] = initial[vector][stage];
            }
        }
        for (int segment_index = 0; segment_index < segment_count - 1; ++segment_index) {
#pragma unroll
            for (int segment_token = 0; segment_token < S; ++segment_token) {
                const int token = first + segment_index * S + segment_token;
                const long long row = static_cast<long long>(batch_index) * tokens + token;
                if (active && (!valid || valid[row])) {
                    // const long long coefficient_offset =
                // (static_cast<long long>(row) * channels + channel) * N;
                    const int base = alpha_slot<N, G>(
                        token - first, local_channel, span);
                    float alpha[M / L][N];
#pragma unroll
                    for (int vector = 0; vector < M / L; ++vector) {
#pragma unroll
                        for (int stage = 0; stage < N; ++stage)
                            alpha[vector][stage] = alpha_cache[base + stage];
                    }
                    const float signal =
                        load_value(x + row * row_stride + channel);
                    float input[M / L];
#pragma unroll
                    for (int vector = 0; vector < M / L; ++vector) {
                        const int memory = lane + vector * L;
                        input[vector] = signal
                                      * write_cache[(token - first) * M + memory];
                    }
                    advance_vectors<N, M / L, MODE>(
                        input, alpha, alpha_cache + base, state);
                }
            }
            if (active) {
#pragma unroll
                for (int vector = 0; vector < M / L; ++vector) {
                    const int memory = lane + vector * L;
#pragma unroll
                    for (int stage = 0; stage < N; ++stage)
                        segment_checkpoint[
                            ((segment_index * G + local_channel) * N + stage) * M
                            + memory] = state[vector][stage];
                }
            }
        }
        __syncthreads();
        float future[M / L][N] = {0.0f};
        if (active) {
#pragma unroll
            for (int vector = 0; vector < M / L; ++vector) {
                const int memory = lane + vector * L;
                load_state<N>(boundary + (plane * M + memory) * N, future[vector]);
            }
        }
        for (int segment_index = segment_count - 1; segment_index >= 0; --segment_index) {
            const int segment_first = first + segment_index * S;
            const int segment_last = segment_first + S < last ? segment_first + S : last;
            float held_state[S][M / L][N];
            if (active) {
#pragma unroll
                for (int vector = 0; vector < M / L; ++vector) {
                    const int memory = lane + vector * L;
                    if (segment_index == 0) {
#pragma unroll
                        for (int stage = 0; stage < N; ++stage)
                            state[vector][stage] = initial[vector][stage];
                    } else {
#pragma unroll
                        for (int stage = 0; stage < N; ++stage)
                            state[vector][stage] = segment_checkpoint[
                                (((segment_index - 1) * G + local_channel) * N + stage) * M
                                + memory];
                    }
                }
            }
            /* THE SEGMENT INDEX IS A COMPILE-TIME CONSTANT so held_state stays in registers; a
             * runtime bound would force it to local memory and spill. S is a template parameter
             * and segment_last is block-uniform -- it is min(segment_first + S, last) and last
             * depends only on chunk_index, chunk and tokens -- so every thread leaves this loop at
             * the same iteration and the exit is not divergent. There is no __syncthreads() in
             * either segment loop, only the __syncwarp() between them. */
#pragma unroll
            for (int segment_token = 0; segment_token < S; ++segment_token) {
                const int token = segment_first + segment_token;
                if (token >= segment_last) break;
                const long long row = static_cast<long long>(batch_index) * tokens + token;
                float levels[M / L][N + 1] = {0.0f};
                const bool token_active = active && (!valid || valid[row]);
                if (token_active) {
#pragma unroll
                    for (int vector = 0; vector < M / L; ++vector) {
#pragma unroll
                        for (int stage = 0; stage < N; ++stage)
                            held_state[segment_token][vector][stage] = state[vector][stage];
                    }
                    // const long long coefficient_offset =
                // (static_cast<long long>(row) * channels + channel) * N;
                    const int base = alpha_slot<N, G>(
                        token - first, local_channel, span);
                    float alpha[M / L][N];
#pragma unroll
                    for (int vector = 0; vector < M / L; ++vector) {
#pragma unroll
                        for (int stage = 0; stage < N; ++stage)
                            alpha[vector][stage] = alpha_cache[base + stage];
                    }
                    const float signal =
                        load_value(x + row * row_stride + channel);
                    float input[M / L];
#pragma unroll
                    for (int vector = 0; vector < M / L; ++vector) {
                        const int memory = lane + vector * L;
                        input[vector] = signal
                                      * write_cache[(token - first) * M + memory];
                    }
                    advance_vectors<N, M / L, MODE>(
                        input, alpha, alpha_cache + base, state, levels);
                }
#pragma unroll
                for (int stage = 0; stage <= N; ++stage) {
                    float sum = 0.0f;
                    if (token_active) {
#pragma unroll
                        for (int vector = 0; vector < M / L; ++vector) {
                            const int memory = lane + vector * L;
                            const float read_value = cache_read
                                ? load_value(read_cache
                                    + ((token - first) * (N + 1) + stage) * M
                                    + memory)
                                : load_read<IO, N, M>(
                                    memory_read, row, read_row_stride, stage, memory);
                            sum = fmaf(levels[vector][stage], read_value, sum);
                        }
                    }
#pragma unroll
                    for (int offset = L / 2; offset; offset >>= 1)
                        sum += __shfl_down_sync(0xffffffffu, sum, offset, L);
                    if (lane == 0)
                        stored_readout[
                            (segment_token * G + local_channel) * (N + 1) + stage] = sum;
                }
            }
            __syncwarp();
#pragma unroll
            for (int segment_token = S - 1; segment_token >= 0; --segment_token) {
                const int token = segment_first + segment_token;
                if (token >= segment_last) continue;
                const int local_token = token - first;
                const long long row = static_cast<long long>(batch_index) * tokens + token;
                const bool token_active = active && (!valid || valid[row]);
                const long long coefficient_offset =
                    row * (packed ? alpha_row_stride : channels * N) + channel * N;
                const long long readout_offset =
                    row * (packed ? rho_row_stride : channels * (N + 1)) + channel * (N + 1);
                const long long rho_offset =
                    row * rho_row_stride + channel * (N + 1);
                const long long alpha_offset =
                    row * alpha_row_stride + channel * N;
                const long long beta_offset =
                    row * beta_row_stride + channel * N;
                float rho[N + 1] = {0.0f};
                float readout[N + 1] = {0.0f};
#pragma unroll
                for (int stage = 0; stage <= N; ++stage) {
                    rho[stage] = token_active
                        ? load_value(raw_rho + rho_offset + stage) : 0.0f;
                    readout[stage] = stored_readout[
                        (segment_token * G + local_channel) * (N + 1) + stage];
                }
                const float raw_gate = token_active
                    ? load_value(gamma + row * gamma_row_stride + channel) : 0.0f;
                const float gate = silu_fast(raw_gate);
                const float go = token_active
                    ? load_value(grad_output + row * channels + channel) : 0.0f;
                float readout_total = 0.0f;
#pragma unroll
                for (int stage = 0; stage <= N; ++stage)
                    readout_total = fmaf(rho[stage], readout[stage], readout_total);
                const float gated = gate;
                if (active && lane == 0) {
#pragma unroll
                    for (int stage = 0; stage <= N; ++stage)
                        store_value(grad_rho + readout_offset + stage,
                                    go * gated * readout[stage]);
                }
                if (active && lane == 0) {
                    const float mixed = readout_total;
                    store_value(grad_gamma + row * channels + channel,
                                go * mixed * silu_grad(raw_gate));
                }
                float alpha_partial[M / L][N] = {0.0f};
                float beta_partial[M / L][N] = {0.0f};
                float signal_partial = 0.0f;
                float write_partial[M / L] = {0.0f};
                float read_partial[N + 1][M / L] = {0.0f};
                if (token_active) {
                    const int base = alpha_slot<N, G>(
                        token - first, local_channel, span);
                    float alpha[M / L][N];
#pragma unroll
                    for (int vector = 0; vector < M / L; ++vector) {
#pragma unroll
                        for (int stage = 0; stage < N; ++stage)
                            alpha[vector][stage] = alpha_cache[base + stage];
                    }
                    const float signal =
                        load_value(x + row * row_stride + channel);
#pragma unroll
                    for (int vector = 0; vector < M / L; ++vector) {
                        const int memory = lane + vector * L;
                        float U[N + 1], V[N + 1], delayed_state[N];
#pragma unroll
                        for (int stage = 0; stage < N; ++stage)
                            delayed_state[stage] = held_state[segment_token][vector][stage];
                        const float write = write_cache[local_token * M + memory];
                        if (MODE == 2) {
                            U[0] = signal * write;
#pragma unroll
                            for (int stage = 0; stage < N; ++stage) {
                                const float delayed = delayed_state[stage];
                                const float cosine = alpha_cache[base + stage];
                                const float sine = alpha_cache[base + N + stage];
                                V[stage + 1] = fmaf(
                                    sine, U[stage], cosine * delayed);
                                U[stage + 1] = fmaf(
                                    cosine, U[stage], -(sine * delayed));
                            }
                            V[0] = U[0];
                        } else {
                        U[N] = signal * write;
#pragma unroll
                        for (int stage = N - 1; stage >= 0; --stage) {
                            const float delayed = delayed_state[stage];
                            if (MODE == 3) {
                                const float scaled_cosine = alpha_cache[base + stage];
                                const float scaled_sine = alpha_cache[base + N + stage];
                                V[stage + 1] = fmaf(
                                    scaled_cosine, delayed, scaled_sine * U[stage + 1]);
                                U[stage] = fmaf(
                                    scaled_cosine, U[stage + 1], -(scaled_sine * delayed));
                            } else if (MODE) {
                                const float cosine = alpha_cache[base + stage];
                                const float sine = alpha_cache[base + N + stage];
                                V[stage + 1] = fmaf(
                                    cosine, delayed, sine * U[stage + 1]);
                                U[stage] = fmaf(
                                    cosine, U[stage + 1], -(sine * delayed));
                            } else {
                                U[stage] = fmaf(
                                    -alpha[vector][stage], delayed, U[stage + 1]);
                                V[stage + 1] = fmaf(
                                    alpha[vector][stage], U[stage], delayed);
                            }
                        }
                        V[0] = U[0];
                        }
                        float level_gradient[N + 1];
#pragma unroll
                        for (int stage = 0; stage <= N; ++stage) {
                            const float read_value = cache_read
                                ? load_value(read_cache
                                    + (local_token * (N + 1) + stage) * M
                                    + memory)
                                : load_read<IO, N, M>(
                                    memory_read, row, read_row_stride, stage, memory);
                            level_gradient[stage] = go * gated * rho[stage] * read_value;
                        }
                        if (MODE == 2) {
#pragma unroll
                            for (int stage = 0; stage < N; ++stage)
                                level_gradient[stage + 1] += future[vector][stage];
                        } else {
#pragma unroll
                        for (int stage = 0; stage < N; ++stage)
                            level_gradient[stage] += future[vector][stage];
                        }
                        float grad_u = level_gradient[0];
                        if (MODE == 2) {
                            grad_u = 0.0f;
#pragma unroll
                            for (int stage = N - 1; stage >= 0; --stage) {
                                const float grad_v = level_gradient[stage + 1];
                                const float cosine = alpha_cache[base + stage];
                                const float sine = alpha_cache[base + N + stage];
                                future[vector][stage] = fmaf(
                                    cosine, grad_v, -(sine * grad_u));
                                // d theta / d alpha = 1 / (1 + alpha^2) = cosine^2, because
                                // cosine = rsqrt(1 + alpha^2) and sine = alpha * cosine, so the
                                // angle gradient on the line above becomes the coefficient
                                // gradient when scaled by cosine^2.
                                alpha_partial[vector][stage] +=
                                    (grad_v * U[stage + 1] - grad_u * V[stage + 1])
                                    * (cosine * cosine);
                                grad_u = fmaf(cosine, grad_u, sine * grad_v);
                            }
                        } else {
#pragma unroll
                        for (int stage = 0; stage < N; ++stage) {
                            const float grad_v = level_gradient[stage + 1];
                            if (MODE == 3) {
                                const float scaled_cosine = alpha_cache[base + stage];
                                const float scaled_sine = alpha_cache[base + N + stage];
                                const float cosine_squared = alpha_cache[base + 2 * N + stage];
                                const float radius_squared = alpha_cache[base + 3 * N + stage];
                                future[vector][stage] = fmaf(
                                    scaled_cosine, grad_v, -(scaled_sine * grad_u));
                                // THE LATTICE STAGE IS r R(theta) AND THE TWO COEFFICIENTS SET IT
                                // SEPARATELY. The cached alpha factor is
                                // d theta / d alpha = -2 / (1 + alpha^2), so multiplying the
                                // rotation's theta derivative gives the coefficient derivative.
                                // r multiplies both outputs, so d U / d beta = U d r / d beta / r
                                // and likewise for V, and d r / d beta = -2 beta r^2 makes that
                                // factor -2 beta r. -beta is applied once at the store.
                                alpha_partial[vector][stage] +=
                                    (grad_v * U[stage] - grad_u * V[stage + 1])
                                    * cosine_squared;
                                beta_partial[vector][stage] +=
                                    (grad_v * V[stage + 1] + grad_u * U[stage])
                                    * radius_squared;
                                grad_u = fmaf(
                                    scaled_cosine, grad_u, scaled_sine * grad_v);
                            } else if (MODE) {
                                const float cosine = alpha_cache[base + stage];
                                const float sine = alpha_cache[base + N + stage];
                                future[vector][stage] = fmaf(
                                    cosine, grad_v, -(sine * grad_u));
                                alpha_partial[vector][stage] +=
                                    grad_v * U[stage] - grad_u * V[stage + 1];
                                grad_u = fmaf(cosine, grad_u, sine * grad_v);
                            } else {
                                grad_u = fmaf(alpha[vector][stage], grad_v, grad_u);
                                future[vector][stage] = fmaf(
                                    -alpha[vector][stage], grad_u, grad_v);
                                alpha_partial[vector][stage] +=
                                    grad_v * U[stage] - grad_u * delayed_state[stage];
                            }
                        }
                        }
                        /* Under mode 2 the stage-zero tap is U_t^(0) itself, so its level gradient
                         * lands on the input alongside the chain's. */
                        const float grad_initial = MODE == 2
                            ? grad_u + level_gradient[0] : grad_u;
                        signal_partial = fmaf(grad_initial, write, signal_partial);
                        write_partial[vector] = grad_initial * signal;
#pragma unroll
                        for (int stage = 0; stage <= N; ++stage)
                            read_partial[stage][vector] =
                                go * gated * rho[stage] * V[stage];
                    }
                }
#pragma unroll
                for (int offset = L / 2; offset; offset >>= 1)
                    signal_partial += __shfl_down_sync(
                        0xffffffffu, signal_partial, offset, L);
                if (active && lane == 0)
                    store_value(grad_x + row * (packed ? row_stride : channels) + channel, signal_partial);
#pragma unroll
                for (int stage = 0; stage < N; ++stage) {
                    float alpha_gradient = 0.0f;
#pragma unroll
                    for (int vector = 0; vector < M / L; ++vector)
                        alpha_gradient += alpha_partial[vector][stage];
#pragma unroll
                    for (int offset = L / 2; offset; offset >>= 1)
                        alpha_gradient += __shfl_down_sync(
                            0xffffffffu, alpha_gradient, offset, L);
                    if (active && lane == 0) {
                        const long long slot = coefficient_offset + stage;
                        const float raw = load_value(raw_alpha + alpha_offset + stage);
                        store_value(grad_alpha + slot,
                                    alpha_gradient
                                    * coefficient_grad());
                    }
                }
                if (MODE == 3) {
#pragma unroll
                    for (int stage = 0; stage < N; ++stage) {
                        float beta_gradient = 0.0f;
#pragma unroll
                        for (int vector = 0; vector < M / L; ++vector)
                            beta_gradient += beta_partial[vector][stage];
#pragma unroll
                        for (int offset = L / 2; offset; offset >>= 1)
                            beta_gradient += __shfl_down_sync(
                                0xffffffffu, beta_gradient, offset, L);
                        if (active && lane == 0) {
                            const long long slot = row
                                * (packed ? beta_row_stride : channels * N) + channel * N + stage;
                            /* the cauchy and gaussian radii cache 2 r and 2 and apply -beta here;
                             * the expexp, expsoftplus and hill radii cache exp(beta),
                             * sigmoid(beta) and -2 beta (1 - r) / max(q, 1e-8)
                             * and apply -1 (stage_radius). */
                            const float beta_factor = radius_of(stage_map) >= 2
                                ? -1.0f : -load_value(raw_beta + beta_offset + stage);
                            store_value(grad_beta + slot, token_active
                                ? beta_factor * beta_gradient : 0.0f);
                        }
                    }
                }
                /* The write gradient is one value per memory lane and the read gradient N+1, so
                 * the cross-channel reduction runs per token over a scratch of N+2 slots per lane
                 * instead of once per segment over two.  Every thread of the block still reduces;
                 * only the number of barriers changes. */
                constexpr int pair_tokens = L == 16 ? (N == 2 ? 4 : 2) : 1;
                constexpr int channel_pairs = L == 16 ? 2 : 1;
                constexpr int reduction_groups = G / channel_pairs;
#pragma unroll
                for (int vector = 0; vector < M / L; ++vector) {
                    if (L == 16) {
                        write_partial[vector] += __shfl_xor_sync(
                            0xffffffffu, write_partial[vector], 16, 32);
#pragma unroll
                        for (int stage = 0; stage <= N; ++stage)
                            read_partial[stage][vector] += __shfl_xor_sync(
                                0xffffffffu, read_partial[stage][vector], 16, 32);
                    }
                    if (local_channel % channel_pairs == 0) {
                        const int memory = lane + vector * L;
                        float *entry = gradient_scratch
                            + ((segment_token % pair_tokens) * reduction_groups
                               + local_channel / channel_pairs) * M * (N + 2);
                        entry[memory] = write_partial[vector];
#pragma unroll
                        for (int stage = 0; stage <= N; ++stage)
                            entry[(stage + 1) * M + memory] = read_partial[stage][vector];
                    }
                }
                if (segment_token % pair_tokens == 0) {
                    __syncthreads();
                    for (int index = threadIdx.x; index < pair_tokens * (N + 2) * M;
                         index += blockDim.x) {
                        const int pair_token = index / ((N + 2) * M);
                        const int coordinate = index % ((N + 2) * M);
                        if (token + pair_token >= segment_last) continue;
                        float value = 0.0f;
#pragma unroll
                        for (int local = 0; local < reduction_groups; ++local)
                            value += gradient_scratch[
                                (pair_token * reduction_groups + local) * M * (N + 2)
                                + coordinate];
                        const int slot = coordinate / M;
                        const int memory = coordinate % M;
                        if (slot == 0)
                            write_accumulator[(local_token + pair_token) * M + memory] += value;
                        else
                            read_accumulator[
                                ((local_token + pair_token) * (N + 1) + slot - 1) * M
                                + memory] += value;
                    }
                    __syncthreads();
                }
            }
            __syncthreads();
        }
    }
    /* grad_memory_groups is [B, T, N+2, M, groups]: slot 0 is the write direction and slots
     * 1..N+1 the read direction of taps 0..N. */
    for (int index = threadIdx.x; index < chunk * M; index += blockDim.x) {
        const int local_token = index / M;
        if (first + local_token < last) {
            const long long row = static_cast<long long>(batch_index) * tokens
                                + first + local_token;
            const long long group_offset =
                static_cast<long long>(group) * grouped_values
                + (row * (N + 2)) * M + (index % M);
            grad_memory_groups[group_offset] = write_accumulator[index];
        }
    }
    for (int index = threadIdx.x; index < chunk * (N + 1) * M; index += blockDim.x) {
        const int local_token = index / ((N + 1) * M);
        if (first + local_token < last) {
            const long long row = static_cast<long long>(batch_index) * tokens
                                + first + local_token;
            const int rest = index % ((N + 1) * M);
            const long long group_offset =
                static_cast<long long>(group) * grouped_values
                + (row * (N + 2) + 1 + rest / M) * M + (rest % M);
            grad_memory_groups[group_offset] = read_accumulator[index];
        }
    }
}

/* One reduction over the channel tiles for both memory directions.  `levels` is K+2 and `memory`
 * is M, so the flat index splits into (row, slot, lane): slot 0 is the write direction and slots
 * 1..K+1 the read direction of taps 0..K, whose gradient is [B, T, K+1, M].  The kernel stays
 * order-agnostic -- the widths are runtime scalars, not template parameters. */
template <typename IO>
__device__ __forceinline__ void backward_phase_four(
    const float *grad_memory_groups,
    IO *grad_write, IO *grad_read, const IO *raw_write,
    int values, int groups, int levels, int memory, int write_stride, int read_stride) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= values) return;
    float total = 0.0f;
    for (int group = 0; group < groups; ++group)
        total += grad_memory_groups[static_cast<long long>(group) * values + index];
    const int lane = index % memory;
    const int slot = (index / memory) % levels;
    const long long row = index / (memory * levels);
    if (slot == 0) {
        if (raw_write) {
            IO rounded;
            store_value(&rounded, total);
            total = load_value(&rounded)
                * silu_grad(load_value(raw_write + row * write_stride + lane));
        }
        store_value(grad_write + row * write_stride + lane, total);
    } else
        store_value(grad_read + row * read_stride + (slot - 1) * memory + lane, total);
}
