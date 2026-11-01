#pragma once
#include "lattice.cuh"

/* ================================================================================================
 * mira_v5_stable: THE INFERENCE SCAN FORMS ITS OWN COEFFICIENTS (alpha, beta, rho) FROM THE CONTROL, WITH A
 * SMALL, BRANCH-FREE TOKEN LOOP.
 *
 * WHY THE CONTROL. alpha, beta and rho are an affine function of one d_r-wide control vector per
 * token,
 *     coefficients = W_coef[rows] . control + b_coef[rows]       (coefficient_projection)
 * and v3 materialises them: 7 d_e = 10,752 bf16 values per token at K = 2, d_e = 1536, written by
 * the projection GEMM and read back by p3i. mira_v4 introduced this kernel: the scan reads the d_r
 * control values instead and forms each (batch, token, channel)'s 2K + K + 1 coefficients on the
 * tensor cores, next to the recurrence.
 *
 * WHAT IS COMPUTED IS v4's, AND SO v3's. Every coefficient is bf16_rn(fp32 sum of d_r exact
 * bf16 x bf16 products + fp32 bias), as the cuBLAS GEMM with its bias epilogue produced it; only
 * the order of the d_r-term sum differs (association). The bf16 rounding is KEPT. The stage map is
 * stage_slots (load_stage_at's body), the recurrence, readout and gate are p3i's (scan_phase_three,
 * INFERENCE, L = 1) instruction for instruction, and memory_write / memory_read are staged widened
 * to float exactly as p3i stages them.
 *
 * WHAT v5 CHANGES (the math is untouched):
 *   1. THE STAGE MAP IS A COMPILE-TIME CONSTANT. v4 passed it at run time, so every token step
 *      carried the instructions of all 15 radius x rotation maps and the branches between them:
 *      the token loop spanned ~14 KB of SASS for ~450 executed instructions. One kernel per map
 *      (mira_fused_scan_k2_m16_r{d_r}_s{mode}_bf16, mode = the launcher's ARCH_STAGE value) keeps
 *      only the selected map, so the hot loop is the recurrence plus a few dozen instructions.
 *   2. THE SEVEN COEFFICIENTS ARE ONE 16-BYTE SHARED LOAD, WITH NO BANK CONFLICT. A (row, channel)'s
 *      eight bf16 columns (alpha N, beta N, rho N + 1, one pad) are one 16-byte chunk, stored at
 *      chunk row * 32 + (channel ^ (row & 7)). The swizzle makes the forming step's bf16x2 stores
 *      conflict-free (its eight g-rows land in eight distinct bank quads instead of one: v4 took
 *      8 wavefronts per store) and the scan's per-token read one LDS.128 over a permutation of one
 *      row's 32 chunks, the minimum 4 wavefronts (v4: three or four loads, each 4-way conflicted).
 *   3. A CYCLE BREAKDOWN. The _timed kernels (d_r = 48) are this kernel with clock64 marks around
 *      each phase; the profiler runs one at each context and records where a warp's cycles go.
 *
 * DECOMPOSITION (closed form, no search). A block is FUSED_BATCH = 8 batch rows (one warp each) x
 * FUSED_CHANNELS = 32 channels (one lane each): 256 threads, two blocks per SM, every thread one
 * (batch, channel) stream of all d_m memory slots, as p3i's l1 map. The block's 32 channels'
 * weights are copied to shared memory once. The token axis is walked in tiles of FUSED_TOKENS = 2,
 * so a tile's (token, batch) rows are exactly one m16 tile. Per tile t:
 *   barrier A (tile t's control and rows are staged; every warp has left tile t - 1's scan);
 *   form    warp w computes channels 4w .. 4w + 3: one m16 tile x 4 n8 tiles x d_r / 16 k-steps
 *           of mma.m16n8k16 (bf16 in, fp32 accumulate), the four channels' chains interleaved,
 *           bias added, rounded once to bf16, into the swizzled shared coefficient tile; and in the
 *           same step stage tile t + 1's control, memory_write and memory_read rows (one 16-byte
 *           chunk per thread, fetched a tile earlier) and fetch tile t + 2's;
 *   barrier B (the coefficient tile is complete);
 *   scan    every thread runs p3i's token step for its stream, with the next token's x and gamma
 *           already in flight.
 * The control tiles and the staged memory_write / memory_read rows are double-buffered; the
 * coefficient tile is single.
 * Grid ceil(B / 8) x ceil(d_e / 32), channel group fastest so a batch group's blocks are adjacent
 * and its rows are read from DRAM once.
 *
 * Inference, one chunk (the launcher's _inference_chunk rule), bf16, sm_80 and later
 * (mma.m16n8k16.bf16). Everything else takes v3's path with the coefficients materialised.
 * ================================================================================================ */
constexpr int FUSED_BATCH = 8;
constexpr int FUSED_CHANNELS = 32;
constexpr int FUSED_TOKENS = 2;
constexpr int FUSED_THREADS = FUSED_BATCH * FUSED_CHANNELS;
constexpr int FUSED_ROWS = FUSED_TOKENS * FUSED_BATCH;             /* = 16, one m16 tile */
constexpr int FUSED_TIMING = 8;                                    /* timed values per warp */
static_assert(FUSED_ROWS == 16, "a tile's (token, batch) rows are one m16 tile");

/* COLUMNS PER CHANNEL: alpha (N), beta (N), rho (N + 1), padded to whole n8 tiles. */
template <int N>
struct FusedColumns {
    static constexpr int used = 3 * N + 1;
    static constexpr int columns = (used + 7) / 8 * 8;
    static constexpr int tiles = columns / 8;
};

/* THE SHARED LAYOUT, IN BYTES: the block's weights, two control tiles, the coefficient tile, then
 * two buffers of the float-staged memory_write / memory_read rows. */
template <int N, int M, int R>
struct FusedShared {
    static constexpr int columns = FusedColumns<N>::columns;
    static constexpr int weights = FUSED_CHANNELS * columns * R * 2;
    static constexpr int control_stride = R + 8;    /* bf16; +8 makes the A loads conflict-free */
    static constexpr int control = FUSED_ROWS * control_stride * 2;
    static constexpr int coefficients = FUSED_ROWS * FUSED_CHANNELS * columns * 2;
    static constexpr int write = FUSED_ROWS * M * 4;
    static constexpr int read = FUSED_ROWS * (N + 1) * M * 4;
    static constexpr int buffer = write + read;
    static constexpr int total = weights + 2 * control + coefficients + 2 * buffer;
    /* 16-byte chunks of one (token, batch) row: control, write, read. */
    static constexpr int control_chunks = R * 2 / 16;
    static constexpr int write_chunks = M * 2 / 16;
    static constexpr int read_chunks = (N + 1) * M * 2 / 16;
    static constexpr int row_chunks = control_chunks + write_chunks + read_chunks;
    static constexpr int chunks = FUSED_ROWS * row_chunks;
};

/* THE SWIZZLED COEFFICIENT CHUNK of tile row `row` ((token, batch) row, 0..15) and block channel
 * `local` (0..31): one 16-byte chunk of the channel's eight bf16 columns. Row r's 32 chunks are a
 * permutation of chunks 32 r .. 32 r + 31, so a warp reading one row touches 512 contiguous bytes;
 * the XOR with r & 7 spreads the forming step's eight rows g = 0..7 over eight bank quads. */
__device__ __forceinline__ int fused_chunk(int row, int local) {
    return row * FUSED_CHANNELS + (local ^ (row & 7));
}

__device__ __forceinline__ void mma_bf16_m16n8k16(
    float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    /* sm_75 has no bf16 mma; the launcher never sends this kernel there (capability < 8.0). */
    __trap();
#endif
}

/* STEP 2: the coefficient tile of one warp's four channels, fragment layouts of the PTX ISA
 * (mma.m16n8k16, .bf16): with g = lane / 4 and q = lane % 4,
 *   A (16 x 16, row): a0 a1 = (g, 2q..2q+1), a2 a3 = (g+8, 2q..), a4 a5 = (g, 2q+8..), a6 a7 = (g+8, 2q+8..)
 *   B (16 x 8, col):  b0 b1 = (k 2q..2q+1, n g), b2 b3 = (k 2q+8..2q+9, n g)
 *   D (16 x 8):       d0 d1 = (g, n 2q..2q+1), d2 d3 = (g+8, n 2q..2q+1)
 * m = token * 8 + batch (so row g is token 0, row g + 8 token 1, both of batch row g), k = the
 * control index, n = the coefficient column of the channel. */
template <int N, int M, int R>
__device__ __forceinline__ void fused_coefficients(
    const __nv_bfloat16 *control_tile, const __nv_bfloat16 *weight_tile, const float *bias,
    __nv_bfloat16 *coefficient_tile, int warp, int lane) {
    constexpr int stride = FusedShared<N, M, R>::control_stride;
    constexpr int columns = FusedColumns<N>::columns;
    constexpr int per_warp = FUSED_CHANNELS / FUSED_BATCH;
    const int g = lane >> 2;
    const int q = lane & 3;
    unsigned a[R / 16][4];
#pragma unroll
    for (int step = 0; step < R / 16; ++step) {
        const int k = step * 16 + 2 * q;
        a[step][0] = *reinterpret_cast<const unsigned *>(control_tile + g * stride + k);
        a[step][1] = *reinterpret_cast<const unsigned *>(control_tile + (g + 8) * stride + k);
        a[step][2] = *reinterpret_cast<const unsigned *>(control_tile + g * stride + k + 8);
        a[step][3] = *reinterpret_cast<const unsigned *>(control_tile + (g + 8) * stride + k + 8);
    }
    /* THE FOUR CHANNELS' CHAINS INTERLEAVED: k-step outer, channel inner, so the warp has four
     * independent accumulators in flight instead of one dependent chain at a time. Each channel's
     * accumulator still adds its k-steps in order 0, 1, ..., R / 16 - 1 (v4's order). */
    static_assert(FusedColumns<N>::tiles == 1, "one n8 tile per channel");
    float d[per_warp][4];
#pragma unroll
    for (int pick = 0; pick < per_warp; ++pick)
#pragma unroll
        for (int i = 0; i < 4; ++i) d[pick][i] = 0.0f;
#pragma unroll
    for (int step = 0; step < R / 16; ++step) {
#pragma unroll
        for (int pick = 0; pick < per_warp; ++pick) {
            const int local = per_warp * warp + pick;
            const __nv_bfloat16 *rows = weight_tile + (local * columns + g) * R;
            const unsigned b[2] = {
                *reinterpret_cast<const unsigned *>(rows + step * 16 + 2 * q),
                *reinterpret_cast<const unsigned *>(rows + step * 16 + 2 * q + 8)};
            mma_bf16_m16n8k16(d[pick], a[step], b);
        }
    }
#pragma unroll
    for (int pick = 0; pick < per_warp; ++pick) {
        const int local = per_warp * warp + pick;
        const int column = 2 * q;
        const float bias_low = bias[local * columns + column];
        const float bias_high = bias[local * columns + column + 1];
        /* bias after the k loop, one rounding: the cuBLAS bias epilogue's order. */
        *reinterpret_cast<__nv_bfloat162 *>(coefficient_tile
            + fused_chunk(g, local) * columns + column) =
            __floats2bfloat162_rn(d[pick][0] + bias_low, d[pick][1] + bias_high);
        *reinterpret_cast<__nv_bfloat162 *>(coefficient_tile
            + fused_chunk(FUSED_BATCH + g, local) * columns + column) =
            __floats2bfloat162_rn(d[pick][2] + bias_low, d[pick][3] + bias_high);
    }
}

/* STEP 1: this thread's 16-byte chunk of tile `tile_first` (zeros past the sequence or the batch). */
template <int N, int M, int R>
__device__ __forceinline__ uint4 fused_fetch(
    const __nv_bfloat16 *__restrict__ control, const __nv_bfloat16 *__restrict__ memory_write,
    const __nv_bfloat16 *__restrict__ memory_read, int batch, int tokens, int batch_group,
    int tile_first, int control_row_stride, int read_row_stride, int chunk) {
    using Shared = FusedShared<N, M, R>;
    uint4 value = make_uint4(0u, 0u, 0u, 0u);
    if (chunk >= Shared::chunks) return value;
    const int row = chunk / Shared::row_chunks;
    const int part = chunk % Shared::row_chunks;
    const int token = tile_first + row / FUSED_BATCH;
    const int sequence = batch_group * FUSED_BATCH + row % FUSED_BATCH;
    if (token >= tokens || sequence >= batch) return value;
    const long long index = static_cast<long long>(sequence) * tokens + token;
    const __nv_bfloat16 *source;
    if (part < Shared::control_chunks)
        source = control + index * control_row_stride + part * 8;
    else if (part < Shared::control_chunks + Shared::write_chunks)
        source = memory_write + index * M + (part - Shared::control_chunks) * 8;
    else
        source = memory_read + index * read_row_stride
               + (part - Shared::control_chunks - Shared::write_chunks) * 8;
    return *reinterpret_cast<const uint4 *>(source);
}

template <int N, int M, int R>
__device__ __forceinline__ void fused_stage(
    __nv_bfloat16 *control_tile, float *cache, uint4 value, int chunk) {
    using Shared = FusedShared<N, M, R>;
    if (chunk >= Shared::chunks) return;
    const int row = chunk / Shared::row_chunks;
    const int part = chunk % Shared::row_chunks;
    if (part < Shared::control_chunks) {
        *reinterpret_cast<uint4 *>(control_tile + row * Shared::control_stride + part * 8) = value;
        return;
    }
    /* memory_write and memory_read are widened here, once per tile, to the floats p3i stages. */
    float *target = part < Shared::control_chunks + Shared::write_chunks
        ? cache + row * M + (part - Shared::control_chunks) * 8
        : cache + FUSED_ROWS * M + row * (N + 1) * M
          + (part - Shared::control_chunks - Shared::write_chunks) * 8;
    const __nv_bfloat16 *pairs = reinterpret_cast<const __nv_bfloat16 *>(&value);
    float4 low, high;
    low.x = load_value(pairs + 0); low.y = load_value(pairs + 1);
    low.z = load_value(pairs + 2); low.w = load_value(pairs + 3);
    high.x = load_value(pairs + 4); high.y = load_value(pairs + 5);
    high.z = load_value(pairs + 6); high.w = load_value(pairs + 7);
    reinterpret_cast<float4 *>(target)[0] = low;
    reinterpret_cast<float4 *>(target)[1] = high;
}

/* mira_dspv2: THE NEXT TOKENS' x AND gamma TRAVEL AS RAW bf16 BITS and are widened where they are
 * used. v4 - v6 widened them at the load (load_value), which put the conversion a few
 * instructions after the LDG; its scoreboard wait stalled every warp for a full global-memory
 * round trip on every token (found by simulator/nvidia from the SASS control codes). The loads
 * now run FUSED_AHEAD tokens ahead and the first wait on them is ~366 instructions later. */
#ifndef FUSED_AHEAD
#define FUSED_AHEAD 2
#endif
__device__ __forceinline__ unsigned short fused_bits(const __nv_bfloat16 *pointer) {
#if defined(__CUDA_ARCH__)
    return __ldg(reinterpret_cast<const unsigned short *>(pointer));
#else
    return *reinterpret_cast<const unsigned short *>(pointer);
#endif
}
__device__ __forceinline__ float fused_widen(unsigned short bits) {
    return __bfloat162float(__ushort_as_bfloat16(bits));
}

/* A CLOCK MARK FOR THE _timed KERNELS: adds the cycles since `mark` to `slot` and moves `mark`. */
__device__ __forceinline__ void fused_mark(long long &mark, long long &slot) {
    const long long now = clock64();
    slot += now - mark;
    mark = now;
}

template <int N, int M, int R, int STAGE, bool TIMED>
__device__ __forceinline__ void fused_scan(
    const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ control,
    const __nv_bfloat16 *__restrict__ weight, const float *__restrict__ bias,
    const __nv_bfloat16 *__restrict__ gamma, const __nv_bfloat16 *__restrict__ memory_write,
    const __nv_bfloat16 *__restrict__ memory_read, __nv_bfloat16 *__restrict__ output,
    const float *__restrict__ initial, float *__restrict__ final_state,
    const bool *__restrict__ valid,
    int batch, int tokens, int channels, int row_stride, int gamma_row_stride,
    int control_row_stride, int read_row_stride, long long *__restrict__ timing) {
    using IO = __nv_bfloat16;
    using Shared = FusedShared<N, M, R>;
    constexpr int L = 1;
    constexpr int columns = FusedColumns<N>::columns;
    static_assert(columns == 8, "a (row, channel)'s coefficients are one 16-byte chunk");
    static_assert(Shared::chunks <= FUSED_THREADS, "one 16-byte chunk per thread per tile");
    static_assert(R % 16 == 0 && M % 8 == 0 && ((N + 1) * M) % 8 == 0, "16-byte rows");
    const int groups = (channels + FUSED_CHANNELS - 1) / FUSED_CHANNELS;
    const int group = static_cast<int>(blockIdx.x % groups);
    const int batch_group = static_cast<int>(blockIdx.x / groups);
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int batch_index = batch_group * FUSED_BATCH + warp;
    const int channel = group * FUSED_CHANNELS + lane;
    const bool active = batch_index < batch && channel < channels;
    /* timed: [0] whole, [1] tile bookkeeping, [2] barrier A, [3] form + stage + fetch,
     * [4] barrier B, [5] next-token loads + coefficient load + stage map,
     * [6] recurrence + readout + gate + store, [7] SM id. */
    long long clocks[FUSED_TIMING - 1] = {0, 0, 0, 0, 0, 0, 0};
    long long mark = 0;
    long long start = 0;
    if (TIMED) { start = clock64(); mark = start; }

    extern __shared__ __align__(16) unsigned char fused_shared[];
    /* THE BLOCK'S WEIGHTS, ONCE: [32 channels][columns][R] bf16 (zeros past d_e), and its bias. */
    IO *const weight_tile = reinterpret_cast<IO *>(fused_shared);
    IO *const control_tiles = reinterpret_cast<IO *>(fused_shared + Shared::weights);
    IO *const coefficient_tile =
        reinterpret_cast<IO *>(fused_shared + Shared::weights + 2 * Shared::control);
    float *const caches = reinterpret_cast<float *>(
        fused_shared + Shared::weights + 2 * Shared::control + Shared::coefficients);
    __shared__ float bias_tile[FUSED_CHANNELS * columns];
    {
        constexpr int per_channel = columns * R / 8;          /* 16-byte chunks */
        for (int index = threadIdx.x; index < FUSED_CHANNELS * per_channel;
             index += FUSED_THREADS) {
            const int local = index / per_channel;
            const int owner = group * FUSED_CHANNELS + local;
            reinterpret_cast<uint4 *>(weight_tile)[index] = owner < channels
                ? reinterpret_cast<const uint4 *>(weight)[
                      static_cast<long long>(owner) * per_channel + index % per_channel]
                : make_uint4(0u, 0u, 0u, 0u);
        }
        for (int index = threadIdx.x; index < FUSED_CHANNELS * columns; index += FUSED_THREADS) {
            const int owner = group * FUSED_CHANNELS + index / columns;
            bias_tile[index] = owner < channels
                ? bias[static_cast<long long>(owner) * columns + index % columns] : 0.0f;
        }
    }

    float state[M / L][N] = {0.0f};
    if (active && initial) {
#pragma unroll
        for (int vector = 0; vector < M / L; ++vector)
#pragma unroll
            for (int stage = 0; stage < N; ++stage)
                state[vector][stage] = initial[
                    ((static_cast<long long>(batch_index) * N + stage) * channels + channel)
                    * M + vector];
    }
    const long long stream_row = static_cast<long long>(batch_index) * tokens;
    const IO *__restrict__ x_cursor = x + stream_row * row_stride + channel;
    const IO *__restrict__ gamma_cursor = gamma + stream_row * gamma_row_stride + channel;
    IO *__restrict__ output_cursor = output + stream_row * channels + channel;
    /* THE NEXT TOKEN'S x AND gamma ARE ALWAYS IN FLIGHT, and the next tile's rows. */
    unsigned short ahead_x[FUSED_AHEAD], ahead_g[FUSED_AHEAD];
#pragma unroll
    for (int k = 0; k < FUSED_AHEAD; ++k) {
        const bool in = active && k < tokens;
        ahead_x[k] = in ? fused_bits(x_cursor + static_cast<long long>(k) * row_stride) : 0;
        ahead_g[k] = in ? fused_bits(gamma_cursor + static_cast<long long>(k) * gamma_row_stride)
                        : 0;
    }
    /* TILE 0 IS STAGED BEFORE THE LOOP; tile t + 1 is staged during tile t's forming step, from
     * a load issued one tile earlier still. */
    constexpr int control_elements = Shared::control / 2;
    constexpr int buffer_floats = Shared::buffer / 4;
    fused_stage<N, M, R>(control_tiles, caches,
                         fused_fetch<N, M, R>(control, memory_write, memory_read, batch, tokens,
                                              batch_group, 0, control_row_stride,
                                              read_row_stride, threadIdx.x),
                         threadIdx.x);
    uint4 pending = make_uint4(0u, 0u, 0u, 0u);
    if (FUSED_TOKENS < tokens)
        pending = fused_fetch<N, M, R>(control, memory_write, memory_read, batch, tokens,
                                       batch_group, FUSED_TOKENS, control_row_stride,
                                       read_row_stride, threadIdx.x);
    int buffer = 0;
    for (int tile_first = 0; tile_first < tokens; tile_first += FUSED_TOKENS) {
        const float *const write_cache = caches + buffer * buffer_floats;
        const float *const read_cache = write_cache + FUSED_ROWS * M;
        if (TIMED) fused_mark(mark, clocks[1]);
        /* A: tile t's control and rows are staged; every warp has left tile t - 1's scan. */
        __syncthreads();
        if (TIMED) fused_mark(mark, clocks[2]);
        /* 1. FORM this warp's four channels' coefficients for tile t, and 2. STAGE tile t + 1
         * (its control tile and row buffer were last read before barrier A) and fetch t + 2. */
        fused_coefficients<N, M, R>(control_tiles + buffer * control_elements, weight_tile,
                                    bias_tile, coefficient_tile, warp, lane);
        if (tile_first + FUSED_TOKENS < tokens) {
            fused_stage<N, M, R>(control_tiles + (buffer ^ 1) * control_elements,
                                 caches + (buffer ^ 1) * buffer_floats, pending, threadIdx.x);
            if (tile_first + 2 * FUSED_TOKENS < tokens)
                pending = fused_fetch<N, M, R>(control, memory_write, memory_read, batch, tokens,
                                               batch_group, tile_first + 2 * FUSED_TOKENS,
                                               control_row_stride, read_row_stride, threadIdx.x);
        }
        if (TIMED) fused_mark(mark, clocks[3]);
        /* B: the coefficient tile is complete. */
        __syncthreads();
        if (TIMED) fused_mark(mark, clocks[4]);
        buffer ^= 1;
        /* 3. SCAN: p3i's token step (scan_phase_three, INFERENCE, L = 1). */
        const int tile_tokens = tile_first + FUSED_TOKENS <= tokens ? FUSED_TOKENS
                                                                    : tokens - tile_first;
        for (int local_token = 0; local_token < tile_tokens; ++local_token) {
            const int token = tile_first + local_token;
            const unsigned short signal_bits = ahead_x[0];
            const unsigned short gate_bits = ahead_g[0];
#pragma unroll
            for (int k = 0; k + 1 < FUSED_AHEAD; ++k) {
                ahead_x[k] = ahead_x[k + 1];
                ahead_g[k] = ahead_g[k + 1];
            }
            if (active && token + FUSED_AHEAD < tokens) {
                ahead_x[FUSED_AHEAD - 1] = fused_bits(
                    x_cursor + static_cast<long long>(FUSED_AHEAD) * row_stride);
                ahead_g[FUSED_AHEAD - 1] = fused_bits(
                    gamma_cursor + static_cast<long long>(FUSED_AHEAD) * gamma_row_stride);
            }
            const int tile_row = local_token * FUSED_BATCH + warp;
            const bool live = !valid || (batch_index < batch && valid[stream_row + token]);
            if (live) {
                if (active) {
                    /* the seven coefficients: one 16-byte chunk, widened as p3i widens them. */
                    const uint4 packed = *reinterpret_cast<const uint4 *>(
                        coefficient_tile + fused_chunk(tile_row, lane) * columns);
                    const IO *values = reinterpret_cast<const IO *>(&packed);
                    float alpha_values[N], beta_values[N], slot[2 * N], rho[N + 1];
#pragma unroll
                    for (int stage = 0; stage < N; ++stage) {
                        alpha_values[stage] = load_value(values + stage);
                        beta_values[stage] = load_value(values + N + stage);
                    }
#pragma unroll
                    for (int stage = 0; stage <= N; ++stage)
                        rho[stage] = load_value(values + 2 * N + stage);
                    stage_slots<N>(alpha_values, beta_values, slot, STAGE);
                    if (TIMED) fused_mark(mark, clocks[5]);
                    const float signal = fused_widen(signal_bits);
                    float pair_total[2] = {0.0f, 0.0f};
                    const float *write_row = write_cache + tile_row * M;
                    const float *read_row = read_cache + tile_row * (N + 1) * M;
#pragma unroll
                    for (int vector = 0; vector < M / L; ++vector) {
                        const int memory = vector;
                        float U = signal * load_value(write_row + memory);
                        float contribution = 0.0f;
#pragma unroll
                        for (int stage = N - 1; stage >= 0; --stage) {
                            const float scaled_cosine = slot[stage];
                            const float scaled_sine = slot[N + stage];
                            const float delayed = state[vector][stage];
                            const float level = fmaf(scaled_cosine, delayed, scaled_sine * U);
                            U = fmaf(scaled_cosine, U, -(scaled_sine * delayed));
                            if (stage + 1 < N) state[vector][stage + 1] = level;
                            contribution = fmaf(
                                rho[stage + 1] * load_value(read_row + (stage + 1) * M + memory),
                                level, contribution);
                        }
                        state[vector][0] = U;
                        contribution = fmaf(rho[0] * load_value(read_row + memory), U,
                                            contribution);
                        pair_total[vector & 1] += contribution;
                    }
                    const float total = pair_total[0] + pair_total[1];
                    store_value(output_cursor, total * silu_fast(fused_widen(gate_bits)));
                    if (TIMED) fused_mark(mark, clocks[6]);
                }
            } else if (active) {
                store_value(output_cursor, 0.0f);
            }
            x_cursor += row_stride;
            gamma_cursor += gamma_row_stride;
            output_cursor += channels;
        }
    }
    if (final_state && active) {
#pragma unroll
        for (int vector = 0; vector < M / L; ++vector)
#pragma unroll
            for (int stage = 0; stage < N; ++stage)
                final_state[
                    ((static_cast<long long>(batch_index) * N + stage) * channels + channel)
                    * M + vector] = state[vector][stage];
    }
    if (TIMED && lane == 0 && timing) {
        clocks[0] = clock64() - start;
        unsigned sm = 0;
#if defined(__CUDA_ARCH__)
        asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
#endif
        long long *row = timing + (static_cast<long long>(blockIdx.x) * FUSED_BATCH + warp)
                         * FUSED_TIMING;
#pragma unroll
        for (int k = 0; k < FUSED_TIMING - 1; ++k) row[k] = clocks[k];
        row[FUSED_TIMING - 1] = sm;
    }
}

/* ONE KERNEL PER (d_r, stage map). MODE is the launcher's ARCH_STAGE value (3 = cauchy_x_mobius,
 * 6 .. 19 the others); the stage map is MODE - 3, a compile-time constant. `mode` is kept in the
 * argument list (v4's) and must equal MODE. */
#define FUSED_SCAN_KERNEL(N, MEMORY, R, BLOCKS, MODE, SUFFIX, TIMED)                           \
extern "C" __global__ __launch_bounds__(FUSED_THREADS, BLOCKS)                                  \
void mira_fused_scan_k##N##_m##MEMORY##_r##R##_s##MODE##_bf16##SUFFIX(                          \
    const __nv_bfloat16 *x, const __nv_bfloat16 *control, const __nv_bfloat16 *weight,          \
    const float *bias, const __nv_bfloat16 *gamma, const __nv_bfloat16 *memory_write,           \
    const __nv_bfloat16 *memory_read, __nv_bfloat16 *output, const float *initial,              \
    float *final_state, const bool *valid, long long *timing,                                   \
    int batch, int tokens, int channels, int row_stride, int gamma_row_stride,                 \
    int control_row_stride, int read_row_stride, int mode) {                                    \
    fused_scan<N, MEMORY, R, MODE - 3, TIMED>(x, control, weight, bias, gamma, memory_write,   \
                                              memory_read, output, initial, final_state, valid, \
                                              batch, tokens, channels, row_stride,              \
                                              gamma_row_stride, control_row_stride,             \
                                              read_row_stride, timing); }

#define FUSED_SCAN_MAPS(N, MEMORY, R, BLOCKS, SUFFIX, TIMED)                                    \
    FUSED_SCAN_KERNEL(N, MEMORY, R, BLOCKS, 3, SUFFIX, TIMED)                                   \
    FUSED_SCAN_KERNEL(N, MEMORY, R, BLOCKS, 6, SUFFIX, TIMED)                                   \
    FUSED_SCAN_KERNEL(N, MEMORY, R, BLOCKS, 7, SUFFIX, TIMED)                                   \
    FUSED_SCAN_KERNEL(N, MEMORY, R, BLOCKS, 8, SUFFIX, TIMED)                                   \
    FUSED_SCAN_KERNEL(N, MEMORY, R, BLOCKS, 9, SUFFIX, TIMED)                                   \
    FUSED_SCAN_KERNEL(N, MEMORY, R, BLOCKS, 10, SUFFIX, TIMED)                                  \
    FUSED_SCAN_KERNEL(N, MEMORY, R, BLOCKS, 11, SUFFIX, TIMED)                                  \
    FUSED_SCAN_KERNEL(N, MEMORY, R, BLOCKS, 12, SUFFIX, TIMED)                                  \
    FUSED_SCAN_KERNEL(N, MEMORY, R, BLOCKS, 13, SUFFIX, TIMED)                                  \
    FUSED_SCAN_KERNEL(N, MEMORY, R, BLOCKS, 14, SUFFIX, TIMED)                                  \
    FUSED_SCAN_KERNEL(N, MEMORY, R, BLOCKS, 15, SUFFIX, TIMED)                                  \
    FUSED_SCAN_KERNEL(N, MEMORY, R, BLOCKS, 16, SUFFIX, TIMED)                                  \
    FUSED_SCAN_KERNEL(N, MEMORY, R, BLOCKS, 17, SUFFIX, TIMED)                                  \
    FUSED_SCAN_KERNEL(N, MEMORY, R, BLOCKS, 18, SUFFIX, TIMED)                                  \
    FUSED_SCAN_KERNEL(N, MEMORY, R, BLOCKS, 19, SUFFIX, TIMED)

// mira_v5_stable: the coefficient-forming inference scan, K = 2, d_m = 16, d_r = 16, 32, 48, 64, one kernel
// per stage map. Two blocks per SM fit the shared block up to d_r = 48; d_r = 64 holds one. The
// _timed kernels (d_r = 48) carry the profiler's clock marks.
