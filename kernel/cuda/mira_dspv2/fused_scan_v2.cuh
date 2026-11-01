#pragma once
#include "fused_scan.cuh"

constexpr int SA_BATCH = 16;
constexpr int SA_HALF = 8;
constexpr int SA_CHANNELS = 32;
constexpr int SA_TOKENS = 4;
constexpr int SA_THREADS = SA_BATCH * SA_CHANNELS;
constexpr int SA_HALF_THREADS = SA_HALF * SA_CHANNELS;
constexpr int SA_ROWS = SA_TOKENS * SA_HALF;
#ifndef SA_AHEAD
#define SA_AHEAD 2
#endif
#ifndef SA_PICKS
#define SA_PICKS 2
#endif

template <int N, int M, int R>
struct SaLayout {
    static constexpr int columns = FusedColumns<N>::columns;
    static constexpr int reads = (N + 1) * M;
    static constexpr int stride = R + 8;
    static constexpr int read_stride = reads + 8;
    static constexpr int weights = SA_CHANNELS * columns * stride * 2;
    static constexpr int read_weights = reads * stride * 2;
    static constexpr int control = SA_ROWS * stride * 2;
    static constexpr int coefficients = SA_ROWS * SA_CHANNELS * columns * 2;
    static constexpr int write = SA_ROWS * M * 4;
    static constexpr int read = SA_ROWS * read_stride * 4;
    static constexpr int half = control + coefficients + 2 * write + read;
    static constexpr int total = weights + read_weights + 2 * half;
    static constexpr int control_chunks = R / 8;
    static constexpr int write_chunks = M / 8;
    static constexpr int row_chunks = control_chunks + write_chunks;
    static constexpr int chunks = SA_ROWS * row_chunks;
    static constexpr int read_tiles = reads / 8;
    static constexpr int m_tiles = SA_ROWS / 16;
    static constexpr int groups = SA_HALF / m_tiles;
    static constexpr int per_group = SA_CHANNELS / groups;
    static_assert(weights % 128 == 0 && read_weights % 128 == 0 && control % 128 == 0
                  && half % 128 == 0, "128-byte aligned regions");
};

__device__ __forceinline__ int sa_chunk(int row, int local) {
    return row * SA_CHANNELS + (local ^ (row & 7));
}

__device__ __forceinline__ void sa_half_sync(int half) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(half + 1), "r"(SA_HALF_THREADS) : "memory");
}

__device__ __forceinline__ unsigned sa_address(const void *pointer) {
    return static_cast<unsigned>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void sa_ldmatrix4(unsigned (&r)[4], unsigned address) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(address));
#else
    __trap();
#endif
}

__device__ __forceinline__ void sa_ldmatrix2(unsigned (&r)[2], unsigned address) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(address));
#else
    __trap();
#endif
}

template <int R>
__device__ __forceinline__ void sa_load_weights(__nv_bfloat16 *tile, const __nv_bfloat16 *source,
                                                int rows, int first, int limit) {
    constexpr int per_row = R / 8;
    for (int index = threadIdx.x; index < rows * per_row; index += SA_THREADS) {
        const int row = index / per_row;
        const int part = index % per_row;
        const int owner = first + row;
        const uint4 value = owner < limit
            ? reinterpret_cast<const uint4 *>(source)[static_cast<long long>(owner) * per_row + part]
            : make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<uint4 *>(tile + row * (R + 8) + part * 8) = value;
    }
}

template <int R, int STEPS>
__device__ __forceinline__ void sa_b_fragments(unsigned (&b)[STEPS][2], unsigned base) {
#pragma unroll
    for (int step = 0; step + 1 < STEPS; step += 2) {
        unsigned r[4];
        sa_ldmatrix4(r, base + step * 32);
        b[step][0] = r[0]; b[step][1] = r[1];
        b[step + 1][0] = r[2]; b[step + 1][1] = r[3];
    }
    if (STEPS % 2) {
        unsigned r[2];
        sa_ldmatrix2(r, base + (STEPS - 1) * 32);
        b[STEPS - 1][0] = r[0]; b[STEPS - 1][1] = r[1];
    }
}

__device__ __forceinline__ void sa_mma(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    __trap();
#endif
}

template <bool V>
struct SaFlag {
    static constexpr bool value = V;
};

struct SaForm {
    unsigned a_at;
    unsigned b_at;
    unsigned read_b_at;
    unsigned coefficient_at;
    unsigned read_at;
    const float *bias;
    const float *read_bias;
    int group;
};

template <int N, int M, int R>
__device__ __forceinline__ SaForm sa_form_plan(unsigned control_base, unsigned weight_base,
                                               unsigned read_base, const float *bias,
                                               const float *read_bias, int warp, int lane) {
    using L = SaLayout<N, M, R>;
    constexpr int row_bytes = L::stride * 2;
    const int tile = warp % L::m_tiles;
    const int group = warp / L::m_tiles;
    const int g = lane >> 2;
    const int q = lane & 3;
    const int row_low = tile * 16 + g;
    const unsigned b_lane = (lane & 7) * row_bytes + ((lane >> 3) & 3) * 16;
    SaForm plan;
    plan.a_at = control_base + (tile * 16 + (lane & 15)) * row_bytes + (lane >> 4) * 16;
    plan.b_at = weight_base + group * L::per_group * L::columns * row_bytes + b_lane;
    plan.read_b_at = read_base + group * 8 * row_bytes + b_lane;
    plan.coefficient_at = ((row_low * SA_CHANNELS + group * L::per_group) * 16 + 4 * q) | (g << 4);
    plan.read_at = (row_low * L::read_stride + group * 8 + 2 * q) * 4;
    plan.bias = bias + group * L::per_group * L::columns + 2 * q;
    plan.read_bias = read_bias + group * 8 + 2 * q;
    plan.group = group;
    return plan;
}

template <int N, int M, int R>
__device__ __forceinline__ void sa_form(const SaForm &plan, unsigned char *coefficient_bytes,
                                        unsigned char *read_bytes) {
    using L = SaLayout<N, M, R>;
    constexpr int columns = L::columns;
    constexpr int steps = R / 16;
    constexpr int row_bytes = L::stride * 2;
    unsigned a[steps][4];
#pragma unroll
    for (int step = 0; step < steps; ++step) sa_ldmatrix4(a[step], plan.a_at + step * 32);
    constexpr int batch_picks = SA_PICKS < L::per_group ? SA_PICKS : L::per_group;
#pragma unroll
    for (int first = 0; first < L::per_group; first += batch_picks) {
        unsigned b[batch_picks][steps][2];
        float d[batch_picks][4];
#pragma unroll
        for (int k = 0; k < batch_picks; ++k) {
            sa_b_fragments<R, steps>(b[k], plan.b_at + (first + k) * columns * row_bytes);
            const float2 bias_pair =
                *reinterpret_cast<const float2 *>(plan.bias + (first + k) * columns);
            d[k][0] = bias_pair.x; d[k][1] = bias_pair.y;
            d[k][2] = bias_pair.x; d[k][3] = bias_pair.y;
        }
#pragma unroll
        for (int step = 0; step < steps; ++step)
#pragma unroll
            for (int k = 0; k < batch_picks; ++k) sa_mma(d[k], a[step], b[k][step]);
#pragma unroll
        for (int k = 0; k < batch_picks; ++k) {
            unsigned char *at = coefficient_bytes + (plan.coefficient_at ^ ((first + k) << 4));
            *reinterpret_cast<__nv_bfloat162 *>(at) = __floats2bfloat162_rn(d[k][0], d[k][1]);
            *reinterpret_cast<__nv_bfloat162 *>(at + 8 * SA_CHANNELS * 16) =
                __floats2bfloat162_rn(d[k][2], d[k][3]);
        }
    }
#pragma unroll
    for (int part = 0; part < (L::read_tiles + L::groups - 1) / L::groups; ++part) {
        if (plan.group + part * L::groups >= L::read_tiles) break;
        unsigned b[steps][2];
        sa_b_fragments<R, steps>(b, plan.read_b_at + part * L::groups * 8 * row_bytes);
        const float2 bias_pair =
            *reinterpret_cast<const float2 *>(plan.read_bias + part * L::groups * 8);
        float d[4] = {bias_pair.x, bias_pair.y, bias_pair.x, bias_pair.y};
#pragma unroll
        for (int step = 0; step < steps; ++step) sa_mma(d, a[step], b[step]);
        const __nv_bfloat162 low = __floats2bfloat162_rn(d[0], d[1]);
        const __nv_bfloat162 high = __floats2bfloat162_rn(d[2], d[3]);
        unsigned char *at = read_bytes + plan.read_at + part * L::groups * 8 * 4;
        *reinterpret_cast<float2 *>(at) = make_float2(__low2float(low), __high2float(low));
        *reinterpret_cast<float2 *>(at + 8 * L::read_stride * 4) =
            make_float2(__low2float(high), __high2float(high));
    }
}

struct SaFetch {
    const __nv_bfloat16 *source;
    long long token_stride;
    int token;
    int target;
    bool live;
    bool control;
};

template <int N, int M, int R>
__device__ __forceinline__ SaFetch sa_fetch_plan(
    const __nv_bfloat16 *control, const __nv_bfloat16 *memory_write, int batch, int tokens,
    int first_sequence, int control_row_stride, int chunk) {
    using L = SaLayout<N, M, R>;
    SaFetch plan;
    const int row = chunk / L::row_chunks;
    const int part = chunk % L::row_chunks;
    const int local_token = row / SA_HALF;
    const int sequence = first_sequence + row % SA_HALF;
    plan.live = chunk < L::chunks && sequence < batch;
    plan.token = local_token;
    plan.control = part < L::control_chunks;
    const long long index = static_cast<long long>(plan.live ? sequence : 0) * tokens + local_token;
    if (plan.control) {
        plan.source = control + index * control_row_stride + part * 8;
        plan.token_stride = control_row_stride;
        plan.target = (row * L::stride + part * 8) * 2;
    } else {
        plan.source = memory_write + index * M + (part - L::control_chunks) * 8;
        plan.token_stride = M;
        plan.target = (row * M + (part - L::control_chunks) * 8) * 4;
    }
    return plan;
}

__device__ __forceinline__ uint4 sa_fetch(const SaFetch &plan, int tile_first, int tokens) {
    uint4 value = make_uint4(0u, 0u, 0u, 0u);
    if (plan.live && tile_first + plan.token < tokens)
        value = __ldg(reinterpret_cast<const uint4 *>(plan.source + tile_first * plan.token_stride));
    return value;
}

template <int N, int M, int R>
__device__ __forceinline__ void sa_stage(const SaFetch &plan, int chunk, unsigned char *control_tile,
                                         unsigned char *write_cache, uint4 value) {
    using L = SaLayout<N, M, R>;
    if (chunk >= L::chunks) return;
    if (plan.control) {
        *reinterpret_cast<uint4 *>(control_tile + plan.target) = value;
        return;
    }
    float4 low, high;
    low.x = __uint_as_float(value.x << 16); low.y = __uint_as_float(value.x & 0xFFFF0000u);
    low.z = __uint_as_float(value.y << 16); low.w = __uint_as_float(value.y & 0xFFFF0000u);
    high.x = __uint_as_float(value.z << 16); high.y = __uint_as_float(value.z & 0xFFFF0000u);
    high.z = __uint_as_float(value.w << 16); high.w = __uint_as_float(value.w & 0xFFFF0000u);
    reinterpret_cast<float4 *>(write_cache + plan.target)[0] = low;
    reinterpret_cast<float4 *>(write_cache + plan.target)[1] = high;
}

__device__ __forceinline__ unsigned short sa_bits(const __nv_bfloat16 *pointer) {
    return __ldg(reinterpret_cast<const unsigned short *>(pointer));
}

template <int M>
__device__ __forceinline__ void sa_row(const float *row, float (&out)[M]) {
#pragma unroll
    for (int k = 0; k < M / 4; ++k) {
        const float4 v = reinterpret_cast<const float4 *>(row)[k];
        out[4 * k] = v.x; out[4 * k + 1] = v.y; out[4 * k + 2] = v.z; out[4 * k + 3] = v.w;
    }
}

template <int N, int M, int R, int STAGE>
__device__ __forceinline__ void sa_scan(
    const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ control,
    const __nv_bfloat16 *__restrict__ weight, const float *__restrict__ bias,
    const __nv_bfloat16 *__restrict__ gamma, const __nv_bfloat16 *__restrict__ memory_write,
    const __nv_bfloat16 *__restrict__ read_weight, const float *__restrict__ read_bias,
    __nv_bfloat16 *__restrict__ output, const float *__restrict__ initial,
    float *__restrict__ final_state, const bool *__restrict__ valid,
    int batch, int tokens, int channels, int row_stride, int gamma_row_stride,
    int control_row_stride) {
    using IO = __nv_bfloat16;
    using L = SaLayout<N, M, R>;
    constexpr int columns = L::columns;
    static_assert(columns == 8, "a (row, channel)'s coefficients are one 16-byte chunk");
    static_assert(L::chunks <= SA_HALF_THREADS, "one 16-byte chunk per thread per tile");
    static_assert(R % 16 == 0 && M % 8 == 0 && L::reads % 8 == 0, "16-byte rows");
    const int groups = (channels + SA_CHANNELS - 1) / SA_CHANNELS;
    const int group = static_cast<int>(blockIdx.x % groups);
    const int batch_group = static_cast<int>(blockIdx.x / groups);
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int half = warp / SA_HALF;
    const int row_warp = warp % SA_HALF;
    const int half_thread = threadIdx.x % SA_HALF_THREADS;
    const int batch_index = batch_group * SA_BATCH + warp;
    const int channel = group * SA_CHANNELS + lane;
    const bool active = batch_index < batch && channel < channels;

    extern __shared__ __align__(16) unsigned char sa_shared[];
    IO *const weight_tile = reinterpret_cast<IO *>(sa_shared);
    IO *const read_tile = reinterpret_cast<IO *>(sa_shared + L::weights);
    unsigned char *const half_base = sa_shared + L::weights + L::read_weights + half * L::half;
    unsigned char *const control_tile = half_base;
    unsigned char *const coefficient_bytes = half_base + L::control;
    unsigned char *const write_caches = coefficient_bytes + L::coefficients;
    unsigned char *const read_bytes = write_caches + 2 * L::write;
    __shared__ float bias_tile[SA_CHANNELS * columns];
    __shared__ float read_bias_tile[L::reads];
    sa_load_weights<R>(weight_tile, weight, SA_CHANNELS * columns, group * SA_CHANNELS * columns,
                       channels * columns);
    sa_load_weights<R>(read_tile, read_weight, L::reads, 0, L::reads);
    for (int index = threadIdx.x; index < SA_CHANNELS * columns; index += SA_THREADS) {
        const int owner = group * SA_CHANNELS + index / columns;
        bias_tile[index] = owner < channels
            ? bias[static_cast<long long>(owner) * columns + index % columns] : 0.0f;
    }
    for (int index = threadIdx.x; index < L::reads; index += SA_THREADS)
        read_bias_tile[index] = read_bias ? read_bias[index] : 0.0f;

    float state[M][N];
#pragma unroll
    for (int vector = 0; vector < M; ++vector)
#pragma unroll
        for (int stage = 0; stage < N; ++stage)
            state[vector][stage] = active && initial
                ? initial[((static_cast<long long>(batch_index) * N + stage) * channels + channel)
                          * M + vector]
                : 0.0f;
    const long long stream_row = static_cast<long long>(active ? batch_index : 0) * tokens;
    const IO *__restrict__ x_cursor = x + (stream_row + SA_AHEAD) * row_stride + channel;
    const IO *__restrict__ gamma_cursor =
        gamma + (stream_row + SA_AHEAD) * gamma_row_stride + channel;
    IO *__restrict__ output_cursor = output + stream_row * channels + channel;
    const bool *__restrict__ valid_cursor = valid ? valid + stream_row : nullptr;
    unsigned short ahead_x[SA_AHEAD], ahead_g[SA_AHEAD];
#pragma unroll
    for (int k = 0; k < SA_AHEAD; ++k) {
        const bool in = active && k < tokens;
        ahead_x[k] = in ? sa_bits(x_cursor + static_cast<long long>(k - SA_AHEAD) * row_stride) : 0;
        ahead_g[k] = in ? sa_bits(gamma_cursor
                                  + static_cast<long long>(k - SA_AHEAD) * gamma_row_stride) : 0;
    }
    const SaFetch plan = sa_fetch_plan<N, M, R>(control, memory_write, batch, tokens,
                                                batch_group * SA_BATCH + half * SA_HALF,
                                                control_row_stride, half_thread);
    int tile_size = half ? SA_TOKENS / 2 : SA_TOKENS;
    sa_stage<N, M, R>(plan, half_thread, control_tile, write_caches, sa_fetch(plan, 0, tokens));
    uint4 pending = sa_fetch(plan, tile_size, tokens);
    const SaForm form = sa_form_plan<N, M, R>(
        sa_address(control_tile), sa_address(weight_tile), sa_address(read_tile), bias_tile,
        read_bias_tile, row_warp, lane);
    const unsigned char *const coefficient_row =
        coefficient_bytes + sa_chunk(row_warp, lane) * columns * 2;
    const int write_row_offset = row_warp * M * 4;
    const unsigned char *const read_row_bytes = read_bytes + row_warp * L::read_stride * 4;
    int buffer = 0;
    __syncthreads();
    for (int tile_first = 0; tile_first < tokens; tile_first += tile_size, tile_size = SA_TOKENS) {
        sa_half_sync(half);
        sa_form<N, M, R>(form, coefficient_bytes, read_bytes);
        sa_half_sync(half);
        const unsigned char *const write_row_bytes =
            write_caches + buffer * L::write + write_row_offset;
        const int next_first = tile_first + tile_size;
        if (next_first < tokens) {
            sa_stage<N, M, R>(plan, half_thread, control_tile, write_caches + (buffer ^ 1) * L::write,
                              pending);
            pending = sa_fetch(plan, next_first + SA_TOKENS, tokens);
        }
        buffer ^= 1;
        const int tile_tokens = next_first <= tokens ? tile_size : tokens - tile_first;
        const auto run = [&](auto has_valid_tag) {
            constexpr bool has_valid = decltype(has_valid_tag)::value;
    #pragma unroll 1
            for (int local_token = 0; local_token < tile_tokens; ++local_token) {
                const int token = tile_first + local_token;
                const unsigned short signal_bits = ahead_x[0];
                const unsigned short gate_bits = ahead_g[0];
    #pragma unroll
                for (int k = 0; k + 1 < SA_AHEAD; ++k) {
                    ahead_x[k] = ahead_x[k + 1];
                    ahead_g[k] = ahead_g[k + 1];
                }
                if (active && token + SA_AHEAD < tokens) {
                    ahead_x[SA_AHEAD - 1] = sa_bits(x_cursor);
                    ahead_g[SA_AHEAD - 1] = sa_bits(gamma_cursor);
                }
                const bool live = !has_valid || valid_cursor[token];
                if (active) {
                    if (live) {
                        const uint4 packed = *reinterpret_cast<const uint4 *>(
                            coefficient_row + local_token * SA_HALF * SA_CHANNELS * 16);
                        const unsigned words[4] = {packed.x, packed.y, packed.z, packed.w};
                        float values[8];
    #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            values[2 * j] = __uint_as_float(words[j] << 16);
                            values[2 * j + 1] = __uint_as_float(words[j] & 0xFFFF0000u);
                        }
                        float alpha_values[N], beta_values[N], slot[2 * N], rho[N + 1];
    #pragma unroll
                        for (int stage = 0; stage < N; ++stage) {
                            alpha_values[stage] = values[stage];
                            beta_values[stage] = values[N + stage];
                        }
    #pragma unroll
                        for (int stage = 0; stage <= N; ++stage) rho[stage] = values[2 * N + stage];
                        stage_slots<N>(alpha_values, beta_values, slot, STAGE);
                        const float signal = __uint_as_float(static_cast<unsigned>(signal_bits) << 16);
                        const float top_cosine = slot[N - 1] * signal;
                        const float top_sine = slot[2 * N - 1] * signal;
                        float write_row[M], read_row[(N + 1) * M];
                        sa_row<M>(reinterpret_cast<const float *>(
                            write_row_bytes + local_token * SA_HALF * M * 4), write_row);
                        sa_row<(N + 1) * M>(reinterpret_cast<const float *>(
                            read_row_bytes + local_token * SA_HALF * L::read_stride * 4), read_row);
                        float acc[N + 1];
    #pragma unroll
                        for (int k = 0; k <= N; ++k) acc[k] = 0.0f;
    #pragma unroll
                        for (int vector = 0; vector < M; ++vector) {
                            const float w = write_row[vector];
                            const float top = state[vector][N - 1];
                            const float top_level = fmaf(slot[N - 1], top, top_sine * w);
                            float U = fmaf(top_cosine, w, -(slot[2 * N - 1] * top));
                            acc[N] = fmaf(read_row[N * M + vector], top_level, acc[N]);
    #pragma unroll
                            for (int stage = N - 2; stage >= 0; --stage) {
                                const float delayed = state[vector][stage];
                                const float level = fmaf(slot[stage], delayed, slot[N + stage] * U);
                                U = fmaf(slot[stage], U, -(slot[N + stage] * delayed));
                                state[vector][stage + 1] = level;
                                acc[stage + 1] = fmaf(read_row[(stage + 1) * M + vector], level,
                                                      acc[stage + 1]);
                            }
                            state[vector][0] = U;
                            acc[0] = fmaf(read_row[vector], U, acc[0]);
                        }
                        float total = 0.0f;
    #pragma unroll
                        for (int k = 0; k <= N; ++k) total = fmaf(rho[k], acc[k], total);
                        const float gate = __uint_as_float(static_cast<unsigned>(gate_bits) << 16);
                        store_value(output_cursor, total * silu_fast(gate));
                    } else {
                        store_value(output_cursor, 0.0f);
                    }
                }
                x_cursor += row_stride;
                gamma_cursor += gamma_row_stride;
                output_cursor += channels;
            }
        };
        if (valid_cursor) run(SaFlag<true>());
        else run(SaFlag<false>());
    }
    if (final_state && active) {
#pragma unroll
        for (int vector = 0; vector < M; ++vector)
#pragma unroll
            for (int stage = 0; stage < N; ++stage)
                final_state[
                    ((static_cast<long long>(batch_index) * N + stage) * channels + channel)
                    * M + vector] = state[vector][stage];
    }
}

#define SA_SCAN_KERNEL(N, MEMORY, R, MODE)                                                      \
extern "C" __global__ __launch_bounds__(SA_THREADS, 1)                                          \
void mira_dspv2_fused_scan_k##N##_m##MEMORY##_r##R##_s##MODE##_bf16(                               \
    const __nv_bfloat16 *x, const __nv_bfloat16 *control, const __nv_bfloat16 *weight,          \
    const float *bias, const __nv_bfloat16 *gamma, const __nv_bfloat16 *memory_write,           \
    const __nv_bfloat16 *read_weight, const float *read_bias, __nv_bfloat16 *output,            \
    const float *initial, float *final_state, const bool *valid,                                \
    int batch, int tokens, int channels, int row_stride, int gamma_row_stride,                 \
    int control_row_stride, int mode) {                                                         \
    sa_scan<N, MEMORY, R, MODE - 3>(x, control, weight, bias, gamma, memory_write, read_weight, \
                                    read_bias, output, initial, final_state, valid, batch,      \
                                    tokens, channels, row_stride, gamma_row_stride,             \
                                    control_row_stride); }

#define SA_SCAN_MAPS(N, MEMORY, R)                                                              \
    SA_SCAN_KERNEL(N, MEMORY, R, 3)                                                             \
    SA_SCAN_KERNEL(N, MEMORY, R, 6)                                                             \
    SA_SCAN_KERNEL(N, MEMORY, R, 7)                                                             \
    SA_SCAN_KERNEL(N, MEMORY, R, 8)                                                             \
    SA_SCAN_KERNEL(N, MEMORY, R, 9)                                                             \
    SA_SCAN_KERNEL(N, MEMORY, R, 10)                                                            \
    SA_SCAN_KERNEL(N, MEMORY, R, 11)                                                            \
    SA_SCAN_KERNEL(N, MEMORY, R, 12)                                                            \
    SA_SCAN_KERNEL(N, MEMORY, R, 13)                                                            \
    SA_SCAN_KERNEL(N, MEMORY, R, 14)                                                            \
    SA_SCAN_KERNEL(N, MEMORY, R, 15)                                                            \
    SA_SCAN_KERNEL(N, MEMORY, R, 16)                                                            \
    SA_SCAN_KERNEL(N, MEMORY, R, 17)                                                            \
    SA_SCAN_KERNEL(N, MEMORY, R, 18)                                                            \
    SA_SCAN_KERNEL(N, MEMORY, R, 19)
