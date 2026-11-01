#pragma once
#include "lattice.cuh"

template <typename IO, int N, int M, int L, int G, int MODE, bool INFERENCE = false>
__device__ __forceinline__ void scan_phase_one(
    const IO *__restrict__ x, const IO *__restrict__ raw_alpha,
    const IO *__restrict__ raw_beta,
    const IO *__restrict__ memory_write,
    float *__restrict__ phi, float *__restrict__ carry,
    float *__restrict__ fine_phi, const bool *__restrict__ valid,
    int batch, int tokens, int channels, int chunk,
    int row_stride, int alpha_row_stride, int beta_row_stride, int fine_chunk,
    int stage_map) {
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
    float basis[N] = {0.0f};
    float aggregate[N] = {0.0f};
    constexpr bool aggregate_in_phi = N > 2 && N <= 4;
    if (N <= 2) {
#pragma unroll
        for (int i = 0; i < N; ++i) {
            const float unit = lane == i ? 1.0f : 0.0f;
            if (!aggregate_in_phi) aggregate[i] = unit;
            basis[i] = unit;
        }
    } else {
        if (!aggregate_in_phi && lane < N) aggregate[lane] = 1.0f;
        if (lane < N) basis[lane] = 1.0f;
    }
    if (aggregate_in_phi && fine_phi && active && lane < N) {
#pragma unroll
        for (int i = 0; i < N; ++i)
            phi[(plane * N + i) * N + lane] = lane == i ? 1.0f : 0.0f;
    }
    /* THE TWO MEMORY DIRECTIONS ARE STAGED IN FLOAT, SO THE CONVERSION HAPPENS ONCE PER BLOCK.
     * memory_write and memory_read carry no channel index, so every one of the G channels in the
     * block reads the same staged value and each converted it from IO at its own point of use --
     * four conversions per thread-token here and twenty-four in p3, for a value the block already
     * holds.  Staging in float moves that conversion into the cooperative load, where it runs once
     * per (block, tile, value).  No global byte moves and no shared access is added: the same
     * store and the same load happen, four bytes wide instead of two, and the reads are
     * four-distinct-address broadcasts so the width introduces no bank conflict.  The value is
     * unchanged, so the arithmetic is bit-identical. */
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
    /* THREE RUNNING POINTERS INSTEAD OF THREE `base + row * stride` CHAINS PER TOKEN.
     *
     * `row` advances by exactly one per token -- the tile loop partitions [first, last) and the
     * inner loop walks each partition contiguously -- so every global stream this loop reads
     * advances by a CONSTANT byte stride.  Recomputing each from `row` cost an IMAD.WIDE.U32
     * plus a LEA.HI.X pair on every token, and the SASS said that class of instruction was
     * 65 percent of p1's token loop while the recurrence itself was 21 percent.  A pointer
     * incremented by its stride is two IADD3.
     *
     * `row` itself stays: `valid[row]` and the fine-phi plane index still need the number.
     *
     * The addresses visited are identical by construction -- p + k*s is the same sequence as
     * base + (r0+k)*s -- and that identity was checked exhaustively over the iteration space
     * before this landed, not argued. */
    const long long first_row = static_cast<long long>(batch_index) * tokens + first;
    const IO *__restrict__ alpha_cursor =
        raw_alpha + first_row * alpha_row_stride + channel * N;
    const IO *__restrict__ beta_cursor =
        raw_beta + first_row * beta_row_stride + channel * N;
    const IO *__restrict__ x_cursor = x + first_row * row_stride + channel;
    /* THE FINE BOUNDARY IS A COUNTDOWN, NOT A MODULO, AND THAT IS WORTH ~20 INSTRUCTIONS A TOKEN.
     *
     * `(token + 1) % fine_chunk == 0` divides by a RUNTIME int. There is no integer-divide unit,
     * so nvcc expands it to the I2F / MUFU.RCP / F2I / IMAD / ISETP fixup sequence -- about twenty
     * instructions, issued at EVERY token, to answer a question that is true once per fine_chunk.
     * The same expansion is paid a second time by `token / fine_chunk` in the body. Together they
     * were the single largest line in this loop's integer mix, and the recurrence they guard is
     * only thirty-eight float instructions.
     *
     * A counter decremented once per token answers it in one instruction, and `fine_index` tracks
     * `token / fine_chunk` by incrementing when the counter resets. Both are seeded from ONE
     * division per block rather than one per token. The predicate and the index are identical to
     * the expressions they replace at every (first, last, fine_chunk) this kernel can be launched
     * with -- that equivalence was checked exhaustively before this landed, not argued. */
    const int fine_chunks = (tokens + fine_chunk - 1) / fine_chunk;
    int fine_countdown = fine_phi ? fine_chunk - (first % fine_chunk) : 1;
    int fine_index = fine_phi ? first / fine_chunk : 0;
    for (int tile_first = first; tile_first < last; tile_first += ALPHA_TILE) {
        const int tile_tokens =
            (tile_first + ALPHA_TILE < last ? ALPHA_TILE : last - tile_first);
        float *const write_cache = tile_base + tile_buffer * stage_tokens * M;
        for (int index = threadIdx.x; index < tile_tokens * M; index += blockDim.x) {
            const int local_token = index / M;
            const int memory = index % M;
            const long long row = static_cast<long long>(batch_index) * tokens
                                + tile_first + local_token;
            write_cache[local_token * M + (memory % L) * (M / L) + memory / L] =
                load_value(memory_write + row * M + memory);
        }
        __syncthreads();
        tile_buffer ^= 1;
        for (int token = tile_first; token < tile_first + tile_tokens; ++token) {
            const long long row = static_cast<long long>(batch_index) * tokens + token;
            const bool live = active && (!valid || valid[row]);
            float slot[2 * N];
            load_stage_at<IO, N>(alpha_cursor, beta_cursor, live, slot, stage_map);
            if (live) {
                /* x is one scalar per channel-token: every lane of the group reads the same
                 * address, which the L1 broadcasts, so the former lane-0 load plus __shfl_sync
                 * is one load. */
                const float signal = load_value(x_cursor);
                float input[M / L];
#pragma unroll
                for (int vector = 0; vector < M / L; ++vector) {
                    const int memory = lane * (M / L) + vector;
                    input[vector] = signal
                        * load_value(write_cache + (token - tile_first) * M + memory);
                }
                advance_vectors<N, M / L, MODE>(input, nullptr, slot, state);
                if (lane < N)
                    advance<N, MODE>(0.0f, slot, basis);
            }
            bool fine_edge = false;
            if (fine_phi) {
                --fine_countdown;
                fine_edge = (fine_countdown == 0) || (token + 1 == last);
            }
            if (fine_edge && active && lane < N) {
                const long long fine_plane = (
                    static_cast<long long>(batch_index) * fine_chunks + fine_index)
                    * channels + channel;
                float next[N] = {0.0f};
                const unsigned mask = __activemask();
#pragma unroll
                for (int row_index = 0; row_index < N; ++row_index) {
                    fine_phi[(fine_plane * N + row_index) * N + lane] = basis[row_index];
#pragma unroll
                    for (int column = 0; column < N; ++column)
                        if (chunk > fine_chunk)
                            next[row_index] = fmaf(
                                __shfl_sync(mask, basis[row_index], column, L),
                                aggregate_in_phi
                                    ? phi[(plane * N + column) * N + lane]
                                    : aggregate[column],
                                next[row_index]);
                    if (chunk <= fine_chunk) next[row_index] = basis[row_index];
                }
#pragma unroll
                for (int i = 0; i < N; ++i) {
                    if (aggregate_in_phi)
                        phi[(plane * N + i) * N + lane] = next[i];
                    else aggregate[i] = next[i];
                    basis[i] = lane == i ? 1.0f : 0.0f;
                }
            }
            /* THE RESET IS OUTSIDE THE STORE because the store is lane- and activity-predicated
             * and the counter is not: every thread in the block walks the same tokens, so every
             * thread must advance the same counter or the next boundary lands on different
             * tokens in different lanes. A counter that reached zero means this token CLOSED its
             * fine chunk, so the next token opens the following one. */
            if (fine_phi && fine_countdown == 0) {
                fine_countdown = fine_chunk;
                ++fine_index;
            }
            alpha_cursor += alpha_row_stride;
            beta_cursor += beta_row_stride;
            x_cursor += row_stride;
        }
    }
    if (active) {
#pragma unroll
        for (int vector = 0; vector < M / L; ++vector) {
            const int memory = lane + vector * L;
            store_state<N>(carry + (plane * M + memory) * N, state[vector]);
        }
    }
    if (active && lane < N && !(aggregate_in_phi && fine_phi)) {
#pragma unroll
        for (int row = 0; row < N; ++row)
            phi[(plane * N + row) * N + lane] = fine_phi ? aggregate[row] : basis[row];
    }
}
