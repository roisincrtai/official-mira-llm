#pragma once
#include "lattice.cuh"

template <typename IO, int N, int M, int L, int G, int MODE, bool INFERENCE = false>
__device__ __forceinline__ void scan_phase_three(
    const IO *__restrict__ x, const IO *__restrict__ raw_alpha,
    const IO *__restrict__ raw_beta,
    const IO *__restrict__ raw_rho, const IO *__restrict__ gamma,
    const IO *__restrict__ memory_write, const IO *__restrict__ memory_read,
    const float *__restrict__ checkpoint,
    IO *__restrict__ output, float *__restrict__ readout_sum,
    float *__restrict__ fine_checkpoint, const float *__restrict__ initial,
    float *__restrict__ final_state, const bool *__restrict__ valid,
    int batch, int tokens, int channels, int chunk,
    int row_stride, int alpha_row_stride, int beta_row_stride,
    int rho_row_stride, int gamma_row_stride, int read_row_stride,
    int fine_chunk, int stage_map) {
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
    const int local_channel = threadIdx.x / L;
    const int channel = group * G + local_channel;
    const bool active = channel < channels;
    const long long plane = (static_cast<long long>(batch_index) * chunks + chunk_index)
                          * channels + channel;
    const int first = chunk_index * chunk;
    const int last = first + chunk < tokens ? first + chunk : tokens;
    float state[M / L][N] = {0.0f};
    if (active) {
#pragma unroll
        for (int vector = 0; vector < M / L; ++vector) {
            const int memory = lane + vector * L;
            if (checkpoint)
                load_state<N>(checkpoint + (plane * M + memory) * N, state[vector]);
            else if (initial)
#pragma unroll
                for (int stage = 0; stage < N; ++stage)
                    state[vector][stage] = initial[
                        ((static_cast<long long>(batch_index) * N + stage) * channels + channel)
                        * M + memory];
        }
    }
    /* Staged in float; see the note in scan_phase_one.  read_cache is the larger half and the one
     * the readout reads K+1 times per memory slot per token, so most of the conversions this
     * removes are p3's. */
    extern __shared__ __align__(16) unsigned char scan_shared[];
    /* THE STAGED TILE IS DOUBLE BUFFERED AND FOLLOWS THE CHUNK.  One barrier a tile is then
     * enough: a thread staging tile i+1 writes the buffer the token loop of tile i does not
     * read, and it cannot reach tile i+2's staging without passing that barrier once more, so
     * the next tile's global loads issue under the current tile's ladder instead of after a
     * block-wide re-convergence.  The tile is never wider than the chunk, so a chunk shorter
     * than ALPHA_TILE -- the single-token decoding chunk is one -- stages its own width, and the
     * launcher sizes the block from the same two expressions. */
    const int stage_tokens = chunk < ALPHA_TILE ? chunk : ALPHA_TILE;
    float *const tile_base = reinterpret_cast<float *>(scan_shared);
    int tile_buffer = 0;
    const long long first_row = static_cast<long long>(batch_index) * tokens + first;
    const IO *__restrict__ alpha_cursor =
        raw_alpha + first_row * alpha_row_stride + channel * N;
    const IO *__restrict__ beta_cursor =
        raw_beta + first_row * beta_row_stride + channel * N;
    const IO *__restrict__ rho_cursor =
        raw_rho + first_row * rho_row_stride + channel * (N + 1);
    const IO *__restrict__ x_cursor = x + first_row * row_stride + channel;
    const IO *__restrict__ gamma_cursor = gamma + first_row * gamma_row_stride + channel;
    IO *__restrict__ output_cursor = output + first_row * channels + channel;
    /* THE FINE BOUNDARY IS A COUNTDOWN HERE TOO; see the note in scan_phase_one. This loop paid
     * the runtime-modulo expansion twice per token -- once for `token % fine_chunk == 0` and once
     * for `token / fine_chunk` in the plane -- against thirty-two float instructions of
     * recurrence, which is why p3's integer mix ran ahead of its arithmetic rather than behind
     * it. p3's predicate is `token % fine_chunk == 0`, so its counter stands ON the boundary
     * rather than one token before it, and is seeded from the chunk's own offset. */
    const int fine_chunks = (tokens + fine_chunk - 1) / fine_chunk;
    /* `fine_left` is the number of tokens still to come BEFORE the next boundary, so it is the
     * distance UP to the next multiple of fine_chunk and not the offset past the last one --
     * seeding it with `first % fine_chunk` inverts the two and silently moves every checkpoint.
     * `fine_index` is the index of that next boundary token, which is what the plane wants. */
    const int fine_offset = fine_checkpoint
        ? (fine_chunk - first % fine_chunk) % fine_chunk : 1;
    int fine_left = fine_offset;
    int fine_index = fine_checkpoint ? (first + fine_offset) / fine_chunk : 0;
    for (int tile_first = first; tile_first < last; tile_first += ALPHA_TILE) {
        const int tile_tokens =
            (tile_first + ALPHA_TILE < last ? ALPHA_TILE : last - tile_first);
        float *const write_cache =
            tile_base + tile_buffer * stage_tokens * (N + 2) * M;
        float *const read_cache = write_cache + stage_tokens * M;
        /* ONE COOPERATIVE LOOP FOR BOTH DIRECTIONS: the read range runs first and the write range
         * follows it, so the block divides tile_tokens * (N + 2) * M values evenly instead of
         * running one loop the block over-covers and a second it under-covers. */
        const int read_values = tile_tokens * (N + 1) * M;
        for (int index = threadIdx.x; index < read_values + tile_tokens * M;
             index += blockDim.x) {
            if (index < read_values) {
                const int local_token = index / ((N + 1) * M);
                const int rest = index % ((N + 1) * M);
                const int stage = rest / M;
                const int memory = rest % M;
                const long long row = static_cast<long long>(batch_index) * tokens
                                    + tile_first + local_token;
                read_cache[(local_token * (N + 1) + stage) * M
                           + (memory % L) * (M / L) + memory / L] = load_value(
                    memory_read + row * read_row_stride + stage * M + memory);
            } else {
                const int rest = index - read_values;
                const int local_token = rest / M;
                const int memory = rest % M;
                const long long row = static_cast<long long>(batch_index) * tokens
                                    + tile_first + local_token;
                write_cache[local_token * M + (memory % L) * (M / L) + memory / L] =
                    load_value(memory_write + row * M + memory);
            }
        }
        __syncthreads();
        tile_buffer ^= 1;
        for (int token = tile_first; token < tile_first + tile_tokens; ++token) {
            const long long row = static_cast<long long>(batch_index) * tokens + token;
            if (fine_checkpoint && fine_left == 0 && active) {  /* standing on the boundary */
                const long long fine_plane = (
                    static_cast<long long>(batch_index) * fine_chunks + fine_index)
                    * channels + channel;
#pragma unroll
                for (int vector = 0; vector < M / L; ++vector)
                    store_state<N>(fine_checkpoint
                        + (fine_plane * M + lane + vector * L) * N, state[vector]);
            }
            /* Advanced by every thread, not only the storing ones; see the note in
             * scan_phase_one. Standing on a boundary reloads the counter, and the token that
             * exhausts it is the last of its fine chunk. */
            if (fine_checkpoint) {
                if (fine_left == 0) {
                    ++fine_index;
                    fine_left = fine_chunk;
                }
                --fine_left;
            }
            const bool live = !valid || valid[row];
            if ((N == 1 || L * G < 512) && !live) {
                if (active && lane == 0) {
                    store_value(INFERENCE ? output_cursor : output + row * channels + channel, 0.0f);
                    if (readout_sum) readout_sum[row * channels + channel] = 0.0f;
                }
                alpha_cursor += alpha_row_stride;
                beta_cursor += beta_row_stride;
                if (INFERENCE) {
                    rho_cursor += rho_row_stride;
                    x_cursor += row_stride;
                    gamma_cursor += gamma_row_stride;
                    output_cursor += channels;
                }
                continue;
            }
            const long long readout_offset =
                row * rho_row_stride + channel * (N + 1);
            float slot[2 * N];
            float rho[N + 1];
            float early_signal = 0.0f;
            float early_gate = 0.0f;
            /* rho carries no activation. */
            /* The ladder ascends and each lattice stage updates its own slot, so the closure does not
             * appear; the stage-zero tap is V_t^(0) = U_t^(0), the current write. */
            if (INFERENCE) {
                load_rho<IO, N, L>(rho_cursor, 0, active && live, rho);
                early_signal = active && live ? load_value(x_cursor) : 0.0f;
                early_gate = active && live ? load_value(gamma_cursor) : 0.0f;
            }
            load_stage_at<IO, N>(alpha_cursor, beta_cursor, active && live, slot, stage_map);
            if (!INFERENCE)
                load_rho<IO, N, L>(raw_rho, readout_offset, active && live, rho);
            float total = 0.0f;
            if (INFERENCE) {
                if (active && live) {
                    const float signal = early_signal;
                    float pair_total[2] = {0.0f, 0.0f};
#pragma unroll
                    for (int vector = 0; vector < M / L; ++vector) {
                        const int memory = lane * (M / L) + vector;
                        float U = signal * load_value(
                            write_cache + (token - tile_first) * M + memory);
                        float contribution = 0.0f;
#pragma unroll
                        for (int stage = N - 1; stage >= 0; --stage) {
                            const float scaled_cosine = slot[stage];
                            const float scaled_sine = slot[N + stage];
                            const float delayed = state[vector][stage];
                            const float level = fmaf(
                                scaled_cosine, delayed, scaled_sine * U);
                            U = fmaf(scaled_cosine, U, -(scaled_sine * delayed));
                            if (stage + 1 < N)
                                state[vector][stage + 1] = level;
                            contribution = fmaf(
                                rho[stage + 1] * load_value(
                                    read_cache
                                    + ((token - tile_first) * (N + 1) + stage + 1) * M
                                    + memory),
                                level, contribution);
                        }
                        state[vector][0] = U;
                        contribution = fmaf(
                            rho[0] * load_value(
                                read_cache + (token - tile_first) * (N + 1) * M + memory),
                            U, contribution);
                        if (L == 1) pair_total[vector & 1] += contribution;
                        else total += contribution;
                    }
                    if (L == 1) total = pair_total[0] + pair_total[1];
                }
            } else {
                float contribution[M / L] = {0.0f};
                if (active && live) {
                    const float signal = load_value(
                        INFERENCE ? x_cursor : x + row * row_stride + channel);
                    float U[M / L];
#pragma unroll
                    for (int vector = 0; vector < M / L; ++vector) {
                        const int memory = lane * (M / L) + vector;
                        U[vector] = signal
                            * load_value(write_cache + (token - tile_first) * M + memory);
                    }
#pragma unroll
                    for (int stage = N - 1; stage >= 0; --stage) {
                        const float scaled_cosine = slot[stage];
                        const float scaled_sine = slot[N + stage];
#pragma unroll
                        for (int vector = 0; vector < M / L; ++vector) {
                            const int memory = lane * (M / L) + vector;
                            const float delayed = state[vector][stage];
                            const float level = fmaf(
                                scaled_cosine, delayed, scaled_sine * U[vector]);
                            U[vector] = fmaf(
                                scaled_cosine, U[vector], -(scaled_sine * delayed));
                            if (stage + 1 < N)
                                state[vector][stage + 1] = level;
                            contribution[vector] = fmaf(
                                rho[stage + 1] * load_value(
                                    read_cache
                                    + ((token - tile_first) * (N + 1) + stage + 1) * M
                                    + memory),
                                level, contribution[vector]);
                        }
                    }
#pragma unroll
                    for (int vector = 0; vector < M / L; ++vector) {
                        const int memory = lane * (M / L) + vector;
                        state[vector][0] = U[vector];
                        contribution[vector] = fmaf(
                            rho[0] * load_value(
                                read_cache + (token - tile_first) * (N + 1) * M + memory),
                            U[vector], contribution[vector]);
                    }
                }
#pragma unroll
                for (int vector = 0; vector < M / L; ++vector)
                    total += contribution[vector];
            }
#pragma unroll
            for (int offset = L / 2; offset; offset >>= 1)
                total += __shfl_down_sync(0xffffffffu, total, offset, L);
            if (active && lane == 0) {
                const float gate = silu_fast(
                    INFERENCE ? early_gate
                              : load_value(gamma + row * gamma_row_stride + channel));
                /* The sum over k is what SiLU is applied to, and what the backward needs before the
                 * gate. b1 walks the reverse scan without replaying the levels, so it cannot form the
                 * sum itself and reads it from here; the pointer is null when no gradient is kept. */
                if (readout_sum != nullptr)
                    readout_sum[row * channels + channel] = total;
                store_value(INFERENCE ? output_cursor : output + row * channels + channel,
                            live ? total * gate : 0.0f);
            }
            alpha_cursor += alpha_row_stride;
            beta_cursor += beta_row_stride;
            if (INFERENCE) {
                rho_cursor += rho_row_stride;
                x_cursor += row_stride;
                gamma_cursor += gamma_row_stride;
                output_cursor += channels;
            }
        }
    }
    if (final_state && active && last == tokens) {
#pragma unroll
        for (int vector = 0; vector < M / L; ++vector)
#pragma unroll
            for (int stage = 0; stage < N; ++stage)
                final_state[
                    ((static_cast<long long>(batch_index) * N + stage) * channels + channel)
                    * M + lane + vector * L] = state[vector][stage];
    }
}
