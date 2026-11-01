#pragma once
#include "common.cuh"

/* mira_v3treescan: THE CHUNK COMPOSITION AS ONE PARALLEL TREE SCAN (replaces v2's p2, p2i, b2 and
 * the 16-chunk affine summary/replay).
 *
 * WHAT IT COMPUTES -- exactly what p2/b2 computed. Chunk j of a stream carries the affine map
 * x -> A_j x + c_j of its zero-carry summary (A_j = Phi_j forward, Phi_j^T backward, read in
 * reverse chunk order backward). The output in the slot of chunk j is the state ENTERING chunk j:
 *     x_0 = initial (forward) or 0 (backward),   x_{j+1} = A_j x_j + c_j,
 * written over c_j (in place, as p2 wrote its checkpoint over the carry). Inference never reads
 * the last chunk's summary, which p1i does not write.
 *
 * DECOMPOSITION. A unit is one (sequence, channel, slice of V memory columns); the state of a unit
 * is N x V floats and every unit is independent. A unit has `workers` threads (a power of two,
 * <= 256, chosen on the host from the GPU), contiguous in the block. Worker w owns the logical
 * chunks [w r, (w + 1) r), r = ceil(chunks / workers):
 *   1. fold    -- composes its chunks into one map (A, c), r steps;
 *   2. scan    -- Kogge-Stone over the workers' maps: shuffles within a warp (log2 min(W, 32)
 *                 steps), then the maps of earlier warps of the unit through shared memory;
 *   3. replay  -- from its exclusive prefix applied to x_0, writes x_j for each of its chunks.
 * DEPTH: 2 r + log2(workers) + (warps per unit) compositions. WORK: 2 chunks reads of (A, c) and
 * one write of x per chunk; N^3 + N^2 V FMAs per fold or scan step, N^2 V per replay step.
 * workers == 1 skips 1 and 2: the replay alone is the serial composition.
 */
template <int N, int V>
__device__ __forceinline__ void tree_identity(float (&a)[N][N], float (&c)[N][V]) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
#pragma unroll
        for (int j = 0; j < N; ++j) a[i][j] = i == j ? 1.0f : 0.0f;
#pragma unroll
        for (int v = 0; v < V; ++v) c[i][v] = 0.0f;
    }
}

/* (a, c) <- (a2, c2) o (a, c): the map that applies (a, c) first and then (a2, c2). */
template <int N, int V>
__device__ __forceinline__ void tree_after(
    float (&a)[N][N], float (&c)[N][V], const float (&a2)[N][N], const float (&c2)[N][V]) {
    float na[N][N], nc[N][V];
#pragma unroll
    for (int i = 0; i < N; ++i) {
#pragma unroll
        for (int j = 0; j < N; ++j) {
            float s = 0.0f;
#pragma unroll
            for (int k = 0; k < N; ++k) s = fmaf(a2[i][k], a[k][j], s);
            na[i][j] = s;
        }
#pragma unroll
        for (int v = 0; v < V; ++v) {
            float s = c2[i][v];
#pragma unroll
            for (int k = 0; k < N; ++k) s = fmaf(a2[i][k], c[k][v], s);
            nc[i][v] = s;
        }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
#pragma unroll
        for (int j = 0; j < N; ++j) a[i][j] = na[i][j];
#pragma unroll
        for (int v = 0; v < V; ++v) c[i][v] = nc[i][v];
    }
}

/* x <- a x + c */
template <int N, int V>
__device__ __forceinline__ void tree_apply(
    float (&x)[N][V], const float (&a)[N][N], const float (&c)[N][V]) {
    float nx[N][V];
#pragma unroll
    for (int i = 0; i < N; ++i)
#pragma unroll
        for (int v = 0; v < V; ++v) {
            float s = c[i][v];
#pragma unroll
            for (int k = 0; k < N; ++k) s = fmaf(a[i][k], x[k][v], s);
            nx[i][v] = s;
        }
#pragma unroll
    for (int i = 0; i < N; ++i)
#pragma unroll
        for (int v = 0; v < V; ++v) x[i][v] = nx[i][v];
}

/* Chunk summary of `plane` for columns [column, column + V): A (transposed backward) and c, in the
 * layouts p1/b1 write: phi[(plane N + row) N + col], carry[(plane M + memory) N + stage]. */
template <int N, int M, int V>
__device__ __forceinline__ void tree_load(
    const float *phi, const float *carry, long long plane, int column, bool transpose,
    float (&a)[N][N], float (&c)[N][V]) {
#pragma unroll
    for (int i = 0; i < N; ++i)
#pragma unroll
        for (int j = 0; j < N; ++j)
            a[i][j] = transpose ? phi[(plane * N + j) * N + i] : phi[(plane * N + i) * N + j];
#pragma unroll
    for (int v = 0; v < V; ++v)
#pragma unroll
        for (int i = 0; i < N; ++i)
            c[i][v] = carry[(plane * M + column + v) * N + i];
}

template <int N, int M, int V>
__device__ __forceinline__ void tree_store(
    float *out, long long plane, int column, const float (&x)[N][V]) {
#pragma unroll
    for (int v = 0; v < V; ++v)
#pragma unroll
        for (int i = 0; i < N; ++i)
            out[(plane * M + column + v) * N + i] = x[i][v];
}

template <int N, int V, int WIDTH>
__device__ __forceinline__ void tree_shuffle_up(
    float (&a)[N][N], float (&c)[N][V], const float (&sa)[N][N], const float (&sc)[N][V],
    int delta) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
#pragma unroll
        for (int j = 0; j < N; ++j) a[i][j] = __shfl_up_sync(0xffffffffu, sa[i][j], delta, WIDTH);
#pragma unroll
        for (int v = 0; v < V; ++v) c[i][v] = __shfl_up_sync(0xffffffffu, sc[i][v], delta, WIDTH);
    }
}

constexpr int TREE_THREADS = 256;
constexpr int TREE_MAX_WARPS = TREE_THREADS / 32;

template <int N, int M, int V>
__device__ __forceinline__ void tree_scan(
    const float *phi, float *carry, const float *initial,
    int batch, int chunks, int channels, int workers, bool reverse, bool inference) {
    constexpr int SLICES = M / V;
    __shared__ float totals[TREE_MAX_WARPS][N * N + N * V];
    const long long thread = static_cast<long long>(blockIdx.x) * TREE_THREADS + threadIdx.x;
    const long long unit = thread / workers;
    const int worker = static_cast<int>(thread % workers);
    const long long units = static_cast<long long>(batch) * channels * SLICES;
    const bool active = unit < units;
    const int slice = static_cast<int>(unit % SLICES);
    const int channel = static_cast<int>((unit / SLICES) % channels);
    const int sequence = static_cast<int>(unit / (static_cast<long long>(SLICES) * channels));
    const int column = slice * V;
    const int per = (chunks + workers - 1) / workers;
    const int first = worker * per;
    const int last = min(chunks, first + per);
    // inference: the last chunk's summary was never written and is never read.
    const int readable = inference ? chunks - 1 : chunks;
    auto plane_of = [&](int logical) {
        const int physical = reverse ? chunks - 1 - logical : logical;
        return (static_cast<long long>(sequence) * chunks + physical) * channels + channel;
    };

    float x[N][V];
#pragma unroll
    for (int i = 0; i < N; ++i)
#pragma unroll
        for (int v = 0; v < V; ++v)
            x[i][v] = active && initial && !reverse
                ? initial[((static_cast<long long>(sequence) * N + i) * channels + channel) * M
                          + column + v]
                : 0.0f;

    if (workers > 1) {
        // 1. fold this worker's chunks.
        float a[N][N], c[N][V];
        tree_identity<N, V>(a, c);
        if (active) {
            for (int logical = first; logical < min(last, readable); ++logical) {
                float la[N][N], lc[N][V];
                tree_load<N, M, V>(phi, carry, plane_of(logical), column, reverse, la, lc);
                tree_after<N, V>(a, c, la, lc);
            }
        }
        // 2a. inclusive scan within the warp's segment of this unit (width min(workers, 32)).
        const int lane = threadIdx.x & 31;
        const int width = workers < 32 ? workers : 32;
        const int position = lane & (width - 1);
        for (int delta = 1; delta < width; delta <<= 1) {
            float pa[N][N], pc[N][V];
            tree_shuffle_up<N, V, 32>(pa, pc, a, c, delta);
            if (position >= delta) {
                // mine applies after the earlier workers' maps
                float ma[N][N], mc[N][V];
#pragma unroll
                for (int i = 0; i < N; ++i) {
#pragma unroll
                    for (int j = 0; j < N; ++j) ma[i][j] = a[i][j];
#pragma unroll
                    for (int v = 0; v < V; ++v) mc[i][v] = c[i][v];
                }
#pragma unroll
                for (int i = 0; i < N; ++i) {
#pragma unroll
                    for (int j = 0; j < N; ++j) a[i][j] = pa[i][j];
#pragma unroll
                    for (int v = 0; v < V; ++v) c[i][v] = pc[i][v];
                }
                tree_after<N, V>(a, c, ma, mc);
            }
        }
        // 2b. the maps of the earlier warps of this unit (workers > 32 only).
        const int warp = threadIdx.x >> 5;
        if (workers > 32) {
            if (lane == 31) {
#pragma unroll
                for (int i = 0; i < N; ++i) {
#pragma unroll
                    for (int j = 0; j < N; ++j) totals[warp][i * N + j] = a[i][j];
#pragma unroll
                    for (int v = 0; v < V; ++v) totals[warp][N * N + i * V + v] = c[i][v];
                }
            }
        }
        __syncthreads();
        // 2c. this worker's exclusive prefix: the inclusive map of the worker before it in the
        // warp, after the maps of the earlier warps.
        float ea[N][N], ec[N][V];
        tree_shuffle_up<N, V, 32>(ea, ec, a, c, 1);
        if (position == 0) tree_identity<N, V>(ea, ec);
        if (workers > 32) {
            const int warps_per_unit = workers / 32;
            const int unit_warp = warp - warp % warps_per_unit;
            float wa[N][N], wc[N][V];
            tree_identity<N, V>(wa, wc);
            for (int earlier = unit_warp; earlier < warp; ++earlier) {
                float ta[N][N], tc[N][V];
#pragma unroll
                for (int i = 0; i < N; ++i) {
#pragma unroll
                    for (int j = 0; j < N; ++j) ta[i][j] = totals[earlier][i * N + j];
#pragma unroll
                    for (int v = 0; v < V; ++v) tc[i][v] = totals[earlier][N * N + i * V + v];
                }
                tree_after<N, V>(wa, wc, ta, tc);
            }
            tree_apply<N, V>(x, wa, wc);
        }
        tree_apply<N, V>(x, ea, ec);
    }
    if (!active) return;
    // 3. replay: the state entering each of this worker's chunks, written over its summary.
    for (int logical = first; logical < last; ++logical) {
        const long long plane = plane_of(logical);
        if (logical >= readable) {
            tree_store<N, M, V>(carry, plane, column, x);
            break;
        }
        float la[N][N], lc[N][V];
        tree_load<N, M, V>(phi, carry, plane, column, reverse, la, lc);
        tree_store<N, M, V>(carry, plane, column, x);
        tree_apply<N, V>(x, la, lc);
    }
}

/* One kernel per (K, d_m); V memory columns per thread (see the instantiations at the end). */
#define TREE_SCAN(N, MEMORY, V)                                                                \
extern "C" __global__ __launch_bounds__(TREE_THREADS) void mira_tree_scan_k##N##_m##MEMORY(  \
    const float *phi, float *carry, const float *initial,                                      \
    int batch, int chunks, int channels, int workers, int reverse, int inference) {          \
    static_assert(MEMORY % V == 0, "the slices must tile the memory axis");                  \
    tree_scan<N, MEMORY, V>(phi, carry, initial, batch, chunks, channels, workers,            \
                            reverse != 0, inference != 0); }
