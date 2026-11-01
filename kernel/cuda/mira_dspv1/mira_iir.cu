#include "common.cuh"
#include "lattice.cuh"
#include "chunk_summary.cuh"
#include "tree_scan.cuh"
#include "chunk_replay.cuh"
#include "backward.cuh"

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 750)
#define MIRA_THREADS_PER_SM 1024
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 870 || __CUDA_ARCH__ == 890 \
                                 || __CUDA_ARCH__ == 1200 || __CUDA_ARCH__ == 1210)
/* 1536 resident threads per SM: Ampere GA10x, Orin, Ada, and consumer Blackwell (12.x, mira_v2). */
#define MIRA_THREADS_PER_SM 1536
#else
#define MIRA_THREADS_PER_SM 2048
#endif
#define MIRA_MIN_BLOCKS(THREADS, N) (((N) <= 2 && (THREADS) * 3 <= MIRA_THREADS_PER_SM) ? 3 : 2)
#define MIRA_INFERENCE_REPLAY_BLOCKS 5

#define PHASE_ONE(NAME, IO, N, MEMORY, LANES, CHANNELS)                                      \
extern "C" __global__ __launch_bounds__(                                                 \
    LANES * CHANNELS, MIRA_MIN_BLOCKS(LANES * CHANNELS, N)) void NAME(                       \
    const IO *x, const IO *raw_alpha, const IO *raw_beta,                                    \
    const IO *memory_write,                                                \
    float *phi, float *carry, float *fine_phi, const bool *valid,                              \
    int batch, int tokens, int channels, int order, int memory, int chunk,   \
    int mode, int row_stride, int alpha_row_stride, int beta_row_stride, int fine_chunk) {                   \
    static_assert(LANES >= N,                                                                \
        "phi is N columns wide and only LANES lanes ever write it: p1 seeds basis[] under "  \
        "(lane < N) and stores phi[(plane*N + row)*N + lane], so LANES < N leaves columns "  \
        "LANES..N-1 of an uninitialised buffer unwritten and makes p2's __shfl_sync(..., "    \
        "column, LANES) wrap them onto lanes 0..LANES-1 -- silently wrong, never an error");  \
    if (order != N || memory != MEMORY) return;                                    \
    scan_phase_one<IO, N, MEMORY, LANES, CHANNELS, 3>(                                   \
        x, raw_alpha, raw_beta, memory_write, phi, carry, fine_phi, valid,                   \
        batch, tokens, channels, chunk, row_stride,                              \
        alpha_row_stride, beta_row_stride, fine_chunk, mode - 3); }

#define PHASE_ONE_INFERENCE(NAME, IO, N, MEMORY, LANES, CHANNELS)                            \
extern "C" __global__ __launch_bounds__(LANES * CHANNELS, 2) void NAME(                     \
    const IO *x, const IO *raw_alpha, const IO *raw_beta,                                    \
    const IO *memory_write, float *phi, float *carry, float *fine_phi, const bool *valid,    \
    int batch, int tokens, int channels, int order, int memory, int chunk,                   \
    int mode, int row_stride, int alpha_row_stride, int beta_row_stride, int fine_chunk) {   \
    static_assert(LANES >= N, "inference scan lanes must cover every phi column");           \
    if (order != N || memory != MEMORY) return;                                               \
    const int chunks = (tokens + chunk - 1) / chunk;                                          \
    const int groups = (channels + CHANNELS - 1) / CHANNELS;                                  \
    const int chunk_index = static_cast<int>((blockIdx.x / groups) % chunks);                 \
    if (chunk_index + 1 == chunks) return;  /* the tree scan never reads it */                                    \
    scan_phase_one<IO, N, MEMORY, LANES, CHANNELS, 3, true>(                                 \
        x, raw_alpha, raw_beta, memory_write, phi, carry, nullptr, valid,                    \
        batch, tokens, channels, chunk, row_stride,                                          \
        alpha_row_stride, beta_row_stride, chunk, mode - 3); }

#define PHASE_THREE(NAME, IO, N, MEMORY, LANES, CHANNELS)                                    \
extern "C" __global__ __launch_bounds__(                                                 \
    LANES * CHANNELS, MIRA_MIN_BLOCKS(LANES * CHANNELS, N)) void NAME(                       \
    const IO *x, const IO *raw_alpha, const IO *raw_beta,                                    \
    const IO *raw_rho, const IO *gamma,                                                      \
    const IO *memory_write, const IO *memory_read,                         \
    const float *checkpoint, IO *output, float *readout_sum, float *fine_checkpoint,            \
    const float *initial, float *final_state, const bool *valid,                                 \
    int batch, int tokens, int channels, int order, int memory, int chunk,   \
    int mode, int row_stride, int alpha_row_stride, int beta_row_stride,                     \
    int rho_row_stride, int gamma_row_stride, int read_row_stride, int fine_chunk) {       \
    static_assert(LANES >= N,                                                                \
        "the K+1 ladder levels are carried one per lane below N: p3 replays the chunk from "  \
        "the checkpoint the l-lane p1/p2 pair wrote, so LANES < N replays a state whose "     \
        "columns LANES..N-1 were never initialised -- silently wrong, never an error");       \
    if (order != N || memory != MEMORY) return;                                    \
    scan_phase_three<IO, N, MEMORY, LANES, CHANNELS, 3>(                                 \
        x, raw_alpha, raw_beta, raw_rho, gamma, memory_write, memory_read,                  \
        checkpoint, output, readout_sum, fine_checkpoint, initial, final_state, valid,                                                     \
        batch, tokens, channels, chunk, row_stride,                              \
        alpha_row_stride, beta_row_stride, rho_row_stride, gamma_row_stride,                 \
        read_row_stride, fine_chunk, mode - 3); }

#define BACKWARD_ONE(NAME, IO, N, MEMORY, LANES, CHANNELS)                                   \
extern "C" __global__ __launch_bounds__(LANES * CHANNELS, 2) void NAME(                    \
    const IO *raw_alpha, const IO *raw_beta, const IO *raw_rho,                             \
    const IO *gamma,                                                                        \
    const IO *memory_read, const IO *grad_output,                          \
    const float *readout_sum, const bool *valid, float *reverse_carry,                      \
    int batch, int tokens, int channels, int order, int memory, int chunk,   \
    int mode, int row_stride, int alpha_row_stride, int beta_row_stride,                     \
    int rho_row_stride, int gamma_row_stride, int read_row_stride) {       \
    static_assert(LANES >= N,                                                                \
        "reverse_phi is N columns wide and only LANES lanes ever write it: b1 repeats p1's "  \
        "basis[N] seed under (lane < N) and stores reverse_phi[(plane*N + row)*N + lane], so " \
        "LANES < N leaves columns LANES..N-1 uninitialised for b2 -- silently wrong gradients"); \
    if (order != N || memory != MEMORY) return;                                    \
    backward_phase_one<IO, N, MEMORY, LANES, CHANNELS, 3>(                               \
        raw_alpha, raw_beta, raw_rho, gamma, memory_read, grad_output,                      \
        readout_sum, valid, reverse_carry,                                             \
        batch, tokens, channels, chunk, row_stride,                              \
        alpha_row_stride, beta_row_stride, rho_row_stride, gamma_row_stride,                 \
        read_row_stride, mode - 3); }

#define PHASE_THREE_INFERENCE(NAME, IO, N, MEMORY, LANES, CHANNELS)                          \
extern "C" __global__ __launch_bounds__(                                                     \
    LANES * CHANNELS, MIRA_INFERENCE_REPLAY_BLOCKS) void NAME(                               \
    const IO *x, const IO *raw_alpha, const IO *raw_beta,                                    \
    const IO *raw_rho, const IO *gamma, const IO *memory_write, const IO *memory_read,       \
    const float *checkpoint, IO *output, float *readout_sum, float *fine_checkpoint,         \
    const float *initial, float *final_state, const bool *valid,                             \
    int batch, int tokens, int channels, int order, int memory, int chunk,                   \
    int mode, int row_stride, int alpha_row_stride, int beta_row_stride,                     \
    int rho_row_stride, int gamma_row_stride, int read_row_stride, int fine_chunk) {         \
    if (order != N || memory != MEMORY) return;                                               \
    scan_phase_three<IO, N, MEMORY, LANES, CHANNELS, 3, true>(                               \
        x, raw_alpha, raw_beta, raw_rho, gamma, memory_write, memory_read,                   \
        checkpoint, output, nullptr, nullptr, initial, final_state, valid,                  \
        batch, tokens, channels, chunk, row_stride, alpha_row_stride,                       \
        beta_row_stride, rho_row_stride, gamma_row_stride, read_row_stride, chunk,          \
        mode - 3); }

// b3 IS INSTANTIATED AT THREE CHANNEL-GROUP WIDTHS PER (K, d_m), NOT TWO, and the third is the
// only one the K = 8 rows can use. Five of the six terms of b3's shared block carry the channel
// group, so the group is the lever that brings K = 8 back under the 49,152 B a block holds
// without the opt-in: at the two widths this file used to carry, (16, 8) asked 71,936 B,
// (32, 8) asked 54,400 B and (64, 8) asked 63,040 B. The third width -- group 4 at d_m 16,
// 2 at d_m 32, 1 at d_m 64 -- brings them to 41,600 B, 48,192 B and 42,784 B. The launcher's
// _backward_variants yields all three for every memory dimension, so every order gets all three
// here and FNS and this file stay in exact correspondence.
#define BACKWARD_THREE(NAME, IO, N, MEMORY, LANES, CHANNELS, SEGMENT)                       \
extern "C" __global__ __launch_bounds__(LANES * CHANNELS, 1) void NAME(                    \
    const IO *x, const IO *raw_alpha, const IO *raw_beta,                                    \
    const IO *raw_rho, const IO *gamma,                                                      \
    const IO *memory_write, const IO *memory_read,                         \
    const float *checkpoint, const float *boundary, const IO *grad_output,                   \
    IO *grad_x, IO *grad_alpha, IO *grad_beta, IO *grad_rho, IO *grad_gamma,                \
    float *grad_memory_groups, const bool *valid,                                                               \
    int batch, int tokens, int channels, int order, int memory, int chunk,   \
    int mode, int groups, int channel_group, int row_stride, int alpha_row_stride,           \
    int beta_row_stride, int rho_row_stride, int gamma_row_stride, int read_row_stride,      \
    int head_channels, int packed) {                                                   \
    if (order != N || memory != MEMORY) return;                                              \
    if (channel_group != CHANNELS) return;                                                   \
    backward_phase_three<IO, N, MEMORY, LANES, CHANNELS, SEGMENT, 3>(                    \
        x, raw_alpha, raw_beta, raw_rho, gamma, memory_write, memory_read,                  \
        checkpoint, boundary,                                                            \
        grad_output, grad_x, grad_alpha, grad_beta, grad_rho, grad_gamma,                \
        grad_memory_groups, valid,                                                       \
        batch, tokens, channels, chunk, groups, channel_group, row_stride,       \
        alpha_row_stride, beta_row_stride, rho_row_stride, gamma_row_stride,                 \
        read_row_stride, head_channels, packed, mode - 3); }

#define BACKWARD_FOUR(NAME, IO)                                                              \
extern "C" __global__ __launch_bounds__(256, 2) void NAME(                                 \
    const float *grad_memory_groups,                                                         \
    IO *grad_write, IO *grad_read, const IO *raw_write,                                      \
    int values, int groups, int levels, int memory, int write_stride, int read_stride) {         \
    backward_phase_four<IO>(                                                                 \
        grad_memory_groups, grad_write, grad_read, raw_write,                                  \
        values, groups, levels, memory, write_stride, read_stride); }

PHASE_ONE(mira_matrix_scan_p1_k1_m8_l2_g256_f32, float, 1, 8, 2, 256)
PHASE_ONE(mira_matrix_scan_p1_k1_m8_l2_g256_bf16, __nv_bfloat16, 1, 8, 2, 256)
PHASE_ONE(mira_matrix_scan_p1_k1_m8_l2_g256_f16, __half, 1, 8, 2, 256)
PHASE_THREE(mira_matrix_scan_p3_k1_m8_l2_g256_f32, float, 1, 8, 2, 256)
PHASE_THREE(mira_matrix_scan_p3_k1_m8_l2_g256_bf16, __nv_bfloat16, 1, 8, 2, 256)
PHASE_THREE(mira_matrix_scan_p3_k1_m8_l2_g256_f16, __half, 1, 8, 2, 256)
BACKWARD_ONE(mira_matrix_scan_b1_k1_m8_l2_g256_f32, float, 1, 8, 2, 256)
BACKWARD_ONE(mira_matrix_scan_b1_k1_m8_l2_g256_bf16, __nv_bfloat16, 1, 8, 2, 256)
BACKWARD_ONE(mira_matrix_scan_b1_k1_m8_l2_g256_f16, __half, 1, 8, 2, 256)
BACKWARD_THREE(mira_matrix_scan_b3_k1_m8_l8_g32_f32, float, 1, 8, 8, 32, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k1_m8_l8_g32_bf16, __nv_bfloat16, 1, 8, 8, 32, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k1_m8_l8_g32_f16, __half, 1, 8, 8, 32, 8)

PHASE_ONE(mira_matrix_scan_p1_k2_m8_l2_g256_f32, float, 2, 8, 2, 256)
PHASE_ONE(mira_matrix_scan_p1_k2_m8_l2_g256_bf16, __nv_bfloat16, 2, 8, 2, 256)
PHASE_ONE(mira_matrix_scan_p1_k2_m8_l2_g256_f16, __half, 2, 8, 2, 256)
PHASE_THREE(mira_matrix_scan_p3_k2_m8_l2_g256_f32, float, 2, 8, 2, 256)
PHASE_THREE(mira_matrix_scan_p3_k2_m8_l2_g256_bf16, __nv_bfloat16, 2, 8, 2, 256)
PHASE_THREE(mira_matrix_scan_p3_k2_m8_l2_g256_f16, __half, 2, 8, 2, 256)
BACKWARD_ONE(mira_matrix_scan_b1_k2_m8_l2_g256_f32, float, 2, 8, 2, 256)
BACKWARD_ONE(mira_matrix_scan_b1_k2_m8_l2_g256_bf16, __nv_bfloat16, 2, 8, 2, 256)
BACKWARD_ONE(mira_matrix_scan_b1_k2_m8_l2_g256_f16, __half, 2, 8, 2, 256)
BACKWARD_THREE(mira_matrix_scan_b3_k2_m8_l8_g16_f32, float, 2, 8, 8, 16, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k2_m8_l8_g16_bf16, __nv_bfloat16, 2, 8, 8, 16, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k2_m8_l8_g16_f16, __half, 2, 8, 8, 16, 8)

PHASE_ONE(mira_matrix_scan_p1_k4_m8_l4_g128_f32, float, 4, 8, 4, 128)
PHASE_ONE(mira_matrix_scan_p1_k4_m8_l4_g128_bf16, __nv_bfloat16, 4, 8, 4, 128)
PHASE_ONE(mira_matrix_scan_p1_k4_m8_l4_g128_f16, __half, 4, 8, 4, 128)
PHASE_THREE(mira_matrix_scan_p3_k4_m8_l4_g128_f32, float, 4, 8, 4, 128)
PHASE_THREE(mira_matrix_scan_p3_k4_m8_l4_g128_bf16, __nv_bfloat16, 4, 8, 4, 128)
PHASE_THREE(mira_matrix_scan_p3_k4_m8_l4_g128_f16, __half, 4, 8, 4, 128)
BACKWARD_ONE(mira_matrix_scan_b1_k4_m8_l4_g128_f32, float, 4, 8, 4, 128)
BACKWARD_ONE(mira_matrix_scan_b1_k4_m8_l4_g128_bf16, __nv_bfloat16, 4, 8, 4, 128)
BACKWARD_ONE(mira_matrix_scan_b1_k4_m8_l4_g128_f16, __half, 4, 8, 4, 128)
BACKWARD_THREE(mira_matrix_scan_b3_k4_m8_l8_g8_f32, float, 4, 8, 8, 8, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k4_m8_l8_g8_bf16, __nv_bfloat16, 4, 8, 8, 8, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k4_m8_l8_g8_f16, __half, 4, 8, 8, 8, 8)

PHASE_ONE(mira_matrix_scan_p1_k8_m8_l8_g64_f32, float, 8, 8, 8, 64)
PHASE_ONE(mira_matrix_scan_p1_k8_m8_l8_g64_bf16, __nv_bfloat16, 8, 8, 8, 64)
PHASE_ONE(mira_matrix_scan_p1_k8_m8_l8_g64_f16, __half, 8, 8, 8, 64)
PHASE_THREE(mira_matrix_scan_p3_k8_m8_l8_g64_f32, float, 8, 8, 8, 64)
PHASE_THREE(mira_matrix_scan_p3_k8_m8_l8_g64_bf16, __nv_bfloat16, 8, 8, 8, 64)
PHASE_THREE(mira_matrix_scan_p3_k8_m8_l8_g64_f16, __half, 8, 8, 8, 64)
BACKWARD_ONE(mira_matrix_scan_b1_k8_m8_l8_g64_f32, float, 8, 8, 8, 64)
BACKWARD_ONE(mira_matrix_scan_b1_k8_m8_l8_g64_bf16, __nv_bfloat16, 8, 8, 8, 64)
BACKWARD_ONE(mira_matrix_scan_b1_k8_m8_l8_g64_f16, __half, 8, 8, 8, 64)
BACKWARD_THREE(mira_matrix_scan_b3_k8_m8_l8_g4_f32, float, 8, 8, 8, 4, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k8_m8_l8_g4_bf16, __nv_bfloat16, 8, 8, 8, 4, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k8_m8_l8_g4_f16, __half, 8, 8, 8, 4, 8)

PHASE_ONE(mira_matrix_scan_p1_k4_m16_l4_g128_f32, float, 4, 16, 4, 128)
PHASE_ONE(mira_matrix_scan_p1_k4_m16_l4_g128_bf16, __nv_bfloat16, 4, 16, 4, 128)
PHASE_ONE(mira_matrix_scan_p1_k4_m16_l4_g128_f16, __half, 4, 16, 4, 128)
PHASE_THREE(mira_matrix_scan_p3_k4_m16_l4_g128_f32, float, 4, 16, 4, 128)
PHASE_THREE(mira_matrix_scan_p3_k4_m16_l4_g128_bf16, __nv_bfloat16, 4, 16, 4, 128)
PHASE_THREE(mira_matrix_scan_p3_k4_m16_l4_g128_f16, __half, 4, 16, 4, 128)
BACKWARD_ONE(mira_matrix_scan_b1_k4_m16_l4_g128_f32, float, 4, 16, 4, 128)
BACKWARD_ONE(mira_matrix_scan_b1_k4_m16_l4_g128_bf16, __nv_bfloat16, 4, 16, 4, 128)
BACKWARD_ONE(mira_matrix_scan_b1_k4_m16_l4_g128_f16, __half, 4, 16, 4, 128)
BACKWARD_THREE(mira_matrix_scan_b3_k4_m16_l16_g16_f32, float, 4, 16, 16, 16, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k4_m16_l16_g16_bf16, __nv_bfloat16, 4, 16, 16, 16, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k4_m16_l16_g16_f16, __half, 4, 16, 16, 16, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k4_m16_l16_g8_f32, float, 4, 16, 16, 8, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k4_m16_l16_g8_bf16, __nv_bfloat16, 4, 16, 16, 8, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k4_m16_l16_g8_f16, __half, 4, 16, 16, 8, 8)

PHASE_ONE(mira_matrix_scan_p1_k4_m32_l8_g64_f32, float, 4, 32, 8, 64)
PHASE_ONE(mira_matrix_scan_p1_k4_m32_l8_g64_bf16, __nv_bfloat16, 4, 32, 8, 64)
PHASE_ONE(mira_matrix_scan_p1_k4_m32_l8_g64_f16, __half, 4, 32, 8, 64)
PHASE_THREE(mira_matrix_scan_p3_k4_m32_l8_g64_f32, float, 4, 32, 8, 64)
PHASE_THREE(mira_matrix_scan_p3_k4_m32_l8_g64_bf16, __nv_bfloat16, 4, 32, 8, 64)
PHASE_THREE(mira_matrix_scan_p3_k4_m32_l8_g64_f16, __half, 4, 32, 8, 64)
BACKWARD_ONE(mira_matrix_scan_b1_k4_m32_l8_g64_f32, float, 4, 32, 8, 64)
BACKWARD_ONE(mira_matrix_scan_b1_k4_m32_l8_g64_bf16, __nv_bfloat16, 4, 32, 8, 64)
BACKWARD_ONE(mira_matrix_scan_b1_k4_m32_l8_g64_f16, __half, 4, 32, 8, 64)
BACKWARD_THREE(mira_matrix_scan_b3_k4_m32_l32_g8_f32, float, 4, 32, 32, 8, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k4_m32_l32_g8_bf16, __nv_bfloat16, 4, 32, 32, 8, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k4_m32_l32_g8_f16, __half, 4, 32, 32, 8, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k4_m32_l32_g4_f32, float, 4, 32, 32, 4, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k4_m32_l32_g4_bf16, __nv_bfloat16, 4, 32, 32, 4, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k4_m32_l32_g4_f16, __half, 4, 32, 32, 4, 8)

PHASE_ONE(mira_matrix_scan_p1_k2_m16_l4_g128_f32, float, 2, 16, 4, 128)
PHASE_ONE(mira_matrix_scan_p1_k2_m16_l4_g128_bf16, __nv_bfloat16, 2, 16, 4, 128)
PHASE_ONE(mira_matrix_scan_p1_k2_m16_l4_g128_f16, __half, 2, 16, 4, 128)
PHASE_ONE_INFERENCE(mira_matrix_scan_p1i_k2_m16_l2_g192_f32, float, 2, 16, 2, 192)
PHASE_ONE_INFERENCE(mira_matrix_scan_p1i_k2_m16_l2_g192_bf16, __nv_bfloat16, 2, 16, 2, 192)
PHASE_ONE_INFERENCE(mira_matrix_scan_p1i_k2_m16_l2_g192_f16, __half, 2, 16, 2, 192)
PHASE_THREE(mira_matrix_scan_p3_k2_m16_l4_g128_f32, float, 2, 16, 4, 128)
PHASE_THREE(mira_matrix_scan_p3_k2_m16_l4_g128_bf16, __nv_bfloat16, 2, 16, 4, 128)
PHASE_THREE(mira_matrix_scan_p3_k2_m16_l4_g128_f16, __half, 2, 16, 4, 128)
PHASE_THREE_INFERENCE(mira_matrix_scan_p3i_k2_m16_l1_g128_f32, float, 2, 16, 1, 128)
PHASE_THREE_INFERENCE(mira_matrix_scan_p3i_k2_m16_l1_g128_bf16, __nv_bfloat16, 2, 16, 1, 128)
PHASE_THREE_INFERENCE(mira_matrix_scan_p3i_k2_m16_l1_g128_f16, __half, 2, 16, 1, 128)
PHASE_ONE_INFERENCE(mira_matrix_scan_p1i_k2_m64_l8_g32_f32, float, 2, 64, 8, 32)
PHASE_ONE_INFERENCE(mira_matrix_scan_p1i_k2_m64_l8_g32_bf16, __nv_bfloat16, 2, 64, 8, 32)
PHASE_ONE_INFERENCE(mira_matrix_scan_p1i_k2_m64_l8_g32_f16, __half, 2, 64, 8, 32)
PHASE_THREE_INFERENCE(mira_matrix_scan_p3i_k2_m64_l4_g32_f32, float, 2, 64, 4, 32)
PHASE_THREE_INFERENCE(mira_matrix_scan_p3i_k2_m64_l4_g32_bf16, __nv_bfloat16, 2, 64, 4, 32)
PHASE_THREE_INFERENCE(mira_matrix_scan_p3i_k2_m64_l4_g32_f16, __half, 2, 64, 4, 32)
BACKWARD_ONE(mira_matrix_scan_b1_k2_m16_l4_g128_f32, float, 2, 16, 4, 128)
BACKWARD_ONE(mira_matrix_scan_b1_k2_m16_l4_g128_bf16, __nv_bfloat16, 2, 16, 4, 128)
BACKWARD_ONE(mira_matrix_scan_b1_k2_m16_l4_g128_f16, __half, 2, 16, 4, 128)
BACKWARD_THREE(mira_matrix_scan_b3_k2_m16_l16_g16_f32, float, 2, 16, 16, 16, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k2_m16_l16_g16_bf16, __nv_bfloat16, 2, 16, 16, 16, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k2_m16_l16_g16_f16, __half, 2, 16, 16, 16, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k2_m16_l16_g8_f32, float, 2, 16, 16, 8, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k2_m16_l16_g8_bf16, __nv_bfloat16, 2, 16, 16, 8, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k2_m16_l16_g8_f16, __half, 2, 16, 16, 8, 8)

PHASE_ONE(mira_matrix_scan_p1_k2_m32_l8_g64_f32, float, 2, 32, 8, 64)
PHASE_ONE(mira_matrix_scan_p1_k2_m32_l8_g64_bf16, __nv_bfloat16, 2, 32, 8, 64)
PHASE_ONE(mira_matrix_scan_p1_k2_m32_l8_g64_f16, __half, 2, 32, 8, 64)
PHASE_THREE(mira_matrix_scan_p3_k2_m32_l8_g64_f32, float, 2, 32, 8, 64)
PHASE_THREE(mira_matrix_scan_p3_k2_m32_l8_g64_bf16, __nv_bfloat16, 2, 32, 8, 64)
PHASE_THREE(mira_matrix_scan_p3_k2_m32_l8_g64_f16, __half, 2, 32, 8, 64)
BACKWARD_ONE(mira_matrix_scan_b1_k2_m32_l8_g64_f32, float, 2, 32, 8, 64)
BACKWARD_ONE(mira_matrix_scan_b1_k2_m32_l8_g64_bf16, __nv_bfloat16, 2, 32, 8, 64)
BACKWARD_ONE(mira_matrix_scan_b1_k2_m32_l8_g64_f16, __half, 2, 32, 8, 64)
BACKWARD_THREE(mira_matrix_scan_b3_k2_m32_l32_g8_f32, float, 2, 32, 32, 8, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k2_m32_l32_g8_bf16, __nv_bfloat16, 2, 32, 32, 8, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k2_m32_l32_g8_f16, __half, 2, 32, 32, 8, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k2_m32_l32_g4_f32, float, 2, 32, 32, 4, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k2_m32_l32_g4_bf16, __nv_bfloat16, 2, 32, 32, 4, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k2_m32_l32_g4_f16, __half, 2, 32, 32, 4, 8)

PHASE_ONE(mira_matrix_scan_p1_k4_m64_l16_g32_f32, float, 4, 64, 16, 32)
PHASE_ONE(mira_matrix_scan_p1_k4_m64_l16_g32_bf16, __nv_bfloat16, 4, 64, 16, 32)
PHASE_ONE(mira_matrix_scan_p1_k4_m64_l16_g32_f16, __half, 4, 64, 16, 32)
PHASE_THREE(mira_matrix_scan_p3_k4_m64_l16_g32_f32, float, 4, 64, 16, 32)
PHASE_THREE(mira_matrix_scan_p3_k4_m64_l16_g32_bf16, __nv_bfloat16, 4, 64, 16, 32)
PHASE_THREE(mira_matrix_scan_p3_k4_m64_l16_g32_f16, __half, 4, 64, 16, 32)
BACKWARD_ONE(mira_matrix_scan_b1_k4_m64_l16_g32_f32, float, 4, 64, 16, 32)
BACKWARD_ONE(mira_matrix_scan_b1_k4_m64_l16_g32_bf16, __nv_bfloat16, 4, 64, 16, 32)
BACKWARD_ONE(mira_matrix_scan_b1_k4_m64_l16_g32_f16, __half, 4, 64, 16, 32)
BACKWARD_THREE(mira_matrix_scan_b3_k4_m64_l32_g4_f32, float, 4, 64, 32, 4, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k4_m64_l32_g4_bf16, __nv_bfloat16, 4, 64, 32, 4, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k4_m64_l32_g4_f16, __half, 4, 64, 32, 4, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k4_m64_l32_g2_f32, float, 4, 64, 32, 2, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k4_m64_l32_g2_bf16, __nv_bfloat16, 4, 64, 32, 2, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k4_m64_l32_g2_f16, __half, 4, 64, 32, 2, 8)

PHASE_ONE(mira_matrix_scan_p1_k2_m64_l16_g32_f32, float, 2, 64, 16, 32)
PHASE_ONE(mira_matrix_scan_p1_k2_m64_l16_g32_bf16, __nv_bfloat16, 2, 64, 16, 32)
PHASE_ONE(mira_matrix_scan_p1_k2_m64_l16_g32_f16, __half, 2, 64, 16, 32)
PHASE_THREE(mira_matrix_scan_p3_k2_m64_l16_g32_f32, float, 2, 64, 16, 32)
PHASE_THREE(mira_matrix_scan_p3_k2_m64_l16_g32_bf16, __nv_bfloat16, 2, 64, 16, 32)
PHASE_THREE(mira_matrix_scan_p3_k2_m64_l16_g32_f16, __half, 2, 64, 16, 32)
BACKWARD_ONE(mira_matrix_scan_b1_k2_m64_l16_g32_f32, float, 2, 64, 16, 32)
BACKWARD_ONE(mira_matrix_scan_b1_k2_m64_l16_g32_bf16, __nv_bfloat16, 2, 64, 16, 32)
BACKWARD_ONE(mira_matrix_scan_b1_k2_m64_l16_g32_f16, __half, 2, 64, 16, 32)
BACKWARD_THREE(mira_matrix_scan_b3_k2_m64_l32_g4_f32, float, 2, 64, 32, 4, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k2_m64_l32_g4_bf16, __nv_bfloat16, 2, 64, 32, 4, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k2_m64_l32_g4_f16, __half, 2, 64, 32, 4, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k2_m64_l32_g2_f32, float, 2, 64, 32, 2, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k2_m64_l32_g2_bf16, __nv_bfloat16, 2, 64, 32, 2, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k2_m64_l32_g2_f16, __half, 2, 64, 32, 2, 8)
PHASE_ONE(mira_matrix_scan_p1_k2_m64_l16_g16_f32, float, 2, 64, 16, 16)
PHASE_ONE(mira_matrix_scan_p1_k2_m64_l16_g16_bf16, __nv_bfloat16, 2, 64, 16, 16)
PHASE_ONE(mira_matrix_scan_p1_k2_m64_l16_g16_f16, __half, 2, 64, 16, 16)
PHASE_ONE(mira_matrix_scan_p1_k2_m64_l16_g24_f32, float, 2, 64, 16, 24)
PHASE_ONE(mira_matrix_scan_p1_k2_m64_l16_g24_bf16, __nv_bfloat16, 2, 64, 16, 24)
PHASE_ONE(mira_matrix_scan_p1_k2_m64_l16_g24_f16, __half, 2, 64, 16, 24)
PHASE_THREE(mira_matrix_scan_p3_k2_m64_l16_g16_f32, float, 2, 64, 16, 16)
PHASE_THREE(mira_matrix_scan_p3_k2_m64_l16_g16_bf16, __nv_bfloat16, 2, 64, 16, 16)
PHASE_THREE(mira_matrix_scan_p3_k2_m64_l16_g16_f16, __half, 2, 64, 16, 16)
BACKWARD_ONE(mira_matrix_scan_b1_k2_m64_l16_g16_f32, float, 2, 64, 16, 16)
BACKWARD_ONE(mira_matrix_scan_b1_k2_m64_l16_g16_bf16, __nv_bfloat16, 2, 64, 16, 16)
BACKWARD_ONE(mira_matrix_scan_b1_k2_m64_l16_g16_f16, __half, 2, 64, 16, 16)
BACKWARD_THREE(mira_matrix_scan_b3_k2_m64_l32_g16_f32, float, 2, 64, 32, 16, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k2_m64_l32_g16_bf16, __nv_bfloat16, 2, 64, 32, 16, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k2_m64_l32_g16_f16, __half, 2, 64, 32, 16, 8)

// ---- lattice_order 1: the ablation sweeps K = 1, 2, 4, 8 (ablation/run.py LATTICE_ORDER_SWEEP) ----
PHASE_ONE(mira_matrix_scan_p1_k1_m16_l4_g128_f32, float, 1, 16, 4, 128)
PHASE_ONE(mira_matrix_scan_p1_k1_m16_l4_g128_bf16, __nv_bfloat16, 1, 16, 4, 128)
PHASE_ONE(mira_matrix_scan_p1_k1_m16_l4_g128_f16, __half, 1, 16, 4, 128)
PHASE_THREE(mira_matrix_scan_p3_k1_m16_l4_g128_f32, float, 1, 16, 4, 128)
PHASE_THREE(mira_matrix_scan_p3_k1_m16_l4_g128_bf16, __nv_bfloat16, 1, 16, 4, 128)
PHASE_THREE(mira_matrix_scan_p3_k1_m16_l4_g128_f16, __half, 1, 16, 4, 128)
BACKWARD_ONE(mira_matrix_scan_b1_k1_m16_l4_g128_f32, float, 1, 16, 4, 128)
BACKWARD_ONE(mira_matrix_scan_b1_k1_m16_l4_g128_bf16, __nv_bfloat16, 1, 16, 4, 128)
BACKWARD_ONE(mira_matrix_scan_b1_k1_m16_l4_g128_f16, __half, 1, 16, 4, 128)
BACKWARD_THREE(mira_matrix_scan_b3_k1_m16_l16_g16_f32, float, 1, 16, 16, 16, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k1_m16_l16_g16_bf16, __nv_bfloat16, 1, 16, 16, 16, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k1_m16_l16_g16_f16, __half, 1, 16, 16, 16, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k1_m16_l16_g8_f32, float, 1, 16, 16, 8, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k1_m16_l16_g8_bf16, __nv_bfloat16, 1, 16, 16, 8, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k1_m16_l16_g8_f16, __half, 1, 16, 16, 8, 8)
// ---- lattice_order 8: the ablation sweeps K = 1, 2, 4, 8 (ablation/run.py LATTICE_ORDER_SWEEP) ----
// THIS IS THE ONE SHAPE WHERE THE LANE FLOOR BINDS. lanes = max(K, d_m / 4) is max(8, 4) = 8
// here, not the 4 that K = 1, 2 and 4 take at d_m = 16, because phi and reverse_phi are eight
// columns wide at K = 8 and only the lanes below N write them; the static_assert(LANES >= N) in
// the macros above rejects the 4 x 128 form these lines used to carry. The block stays 512
// threads, now 8 lanes x 64 channels.
PHASE_ONE(mira_matrix_scan_p1_k8_m16_l8_g64_f32, float, 8, 16, 8, 64)
PHASE_ONE(mira_matrix_scan_p1_k8_m16_l8_g64_bf16, __nv_bfloat16, 8, 16, 8, 64)
PHASE_ONE(mira_matrix_scan_p1_k8_m16_l8_g64_f16, __half, 8, 16, 8, 64)
PHASE_THREE(mira_matrix_scan_p3_k8_m16_l8_g64_f32, float, 8, 16, 8, 64)
PHASE_THREE(mira_matrix_scan_p3_k8_m16_l8_g64_bf16, __nv_bfloat16, 8, 16, 8, 64)
PHASE_THREE(mira_matrix_scan_p3_k8_m16_l8_g64_f16, __half, 8, 16, 8, 64)
BACKWARD_ONE(mira_matrix_scan_b1_k8_m16_l8_g64_f32, float, 8, 16, 8, 64)
BACKWARD_ONE(mira_matrix_scan_b1_k8_m16_l8_g64_bf16, __nv_bfloat16, 8, 16, 8, 64)
BACKWARD_ONE(mira_matrix_scan_b1_k8_m16_l8_g64_f16, __half, 8, 16, 8, 64)

BACKWARD_THREE(mira_matrix_scan_b3_k8_m16_l16_g16_f32, float, 8, 16, 16, 16, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k8_m16_l16_g16_bf16, __nv_bfloat16, 8, 16, 16, 16, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k8_m16_l16_g16_f16, __half, 8, 16, 16, 16, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k8_m16_l16_g8_f32, float, 8, 16, 16, 8, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k8_m16_l16_g8_bf16, __nv_bfloat16, 8, 16, 16, 8, 8)
BACKWARD_THREE(mira_matrix_scan_b3_k8_m16_l16_g8_f16, __half, 8, 16, 16, 8, 8)
// b4 is the channel-tile reduction and does not read K; the k4 in its name is historical.
BACKWARD_FOUR(mira_matrix_scan_b4_k4_f32, float)
BACKWARD_FOUR(mira_matrix_scan_b4_k4_bf16, __nv_bfloat16)
BACKWARD_FOUR(mira_matrix_scan_b4_k4_f16, __half)

PHASE_THREE(mira_matrix_step_k1_m8_f32, float, 1, 8, 2, 32)
PHASE_THREE(mira_matrix_step_k1_m8_bf16, __nv_bfloat16, 1, 8, 2, 32)
PHASE_THREE(mira_matrix_step_k1_m8_f16, __half, 1, 8, 2, 32)
PHASE_THREE(mira_matrix_step_k2_m8_f32, float, 2, 8, 2, 32)
PHASE_THREE(mira_matrix_step_k2_m8_bf16, __nv_bfloat16, 2, 8, 2, 32)
PHASE_THREE(mira_matrix_step_k2_m8_f16, __half, 2, 8, 2, 32)
PHASE_THREE(mira_matrix_step_k4_m8_f32, float, 4, 8, 4, 16)
PHASE_THREE(mira_matrix_step_k4_m8_bf16, __nv_bfloat16, 4, 8, 4, 16)
PHASE_THREE(mira_matrix_step_k4_m8_f16, __half, 4, 8, 4, 16)
PHASE_THREE(mira_matrix_step_k8_m8_f32, float, 8, 8, 8, 8)
PHASE_THREE(mira_matrix_step_k8_m8_bf16, __nv_bfloat16, 8, 8, 8, 8)
PHASE_THREE(mira_matrix_step_k8_m8_f16, __half, 8, 8, 8, 8)
PHASE_THREE(mira_matrix_step_k1_m16_f32, float, 1, 16, 4, 16)
PHASE_THREE(mira_matrix_step_k1_m16_bf16, __nv_bfloat16, 1, 16, 4, 16)
PHASE_THREE(mira_matrix_step_k1_m16_f16, __half, 1, 16, 4, 16)
PHASE_THREE(mira_matrix_step_k2_m16_f32, float, 2, 16, 4, 16)
PHASE_THREE(mira_matrix_step_k2_m16_bf16, __nv_bfloat16, 2, 16, 4, 16)
PHASE_THREE(mira_matrix_step_k2_m16_f16, __half, 2, 16, 4, 16)
PHASE_THREE(mira_matrix_step_k2_m32_f32, float, 2, 32, 8, 8)
PHASE_THREE(mira_matrix_step_k2_m32_bf16, __nv_bfloat16, 2, 32, 8, 8)
PHASE_THREE(mira_matrix_step_k2_m32_f16, __half, 2, 32, 8, 8)
PHASE_THREE(mira_matrix_step_k2_m64_f32, float, 2, 64, 16, 4)
PHASE_THREE(mira_matrix_step_k2_m64_bf16, __nv_bfloat16, 2, 64, 16, 4)
PHASE_THREE(mira_matrix_step_k2_m64_f16, __half, 2, 64, 16, 4)
PHASE_THREE(mira_matrix_step_k4_m16_f32, float, 4, 16, 4, 16)
PHASE_THREE(mira_matrix_step_k4_m16_bf16, __nv_bfloat16, 4, 16, 4, 16)
PHASE_THREE(mira_matrix_step_k4_m16_f16, __half, 4, 16, 4, 16)
PHASE_THREE(mira_matrix_step_k4_m32_f32, float, 4, 32, 8, 8)
PHASE_THREE(mira_matrix_step_k4_m32_bf16, __nv_bfloat16, 4, 32, 8, 8)
PHASE_THREE(mira_matrix_step_k4_m32_f16, __half, 4, 32, 8, 8)
PHASE_THREE(mira_matrix_step_k4_m64_f32, float, 4, 64, 16, 4)
PHASE_THREE(mira_matrix_step_k4_m64_bf16, __nv_bfloat16, 4, 64, 16, 4)
PHASE_THREE(mira_matrix_step_k4_m64_f16, __half, 4, 64, 16, 4)
PHASE_THREE(mira_matrix_step_k8_m16_f32, float, 8, 16, 8, 8)
PHASE_THREE(mira_matrix_step_k8_m16_bf16, __nv_bfloat16, 8, 16, 8, 8)
PHASE_THREE(mira_matrix_step_k8_m16_f16, __half, 8, 16, 8, 8)

// mira_v3treescan: the chunk composition, one kernel per built (K, d_m), V = min(d_m, 8 / K) memory
// columns per thread: K V <= 8 keeps the unit's state and its map in registers without spills at
// every order (sm_90: 48, 64, 100 and 255 registers at K = 1, 2, 4, 8).
TREE_SCAN(1, 8, 8)
TREE_SCAN(1, 16, 8)
TREE_SCAN(2, 8, 4)
TREE_SCAN(2, 16, 4)
TREE_SCAN(2, 32, 4)
TREE_SCAN(2, 64, 4)
TREE_SCAN(4, 8, 2)
TREE_SCAN(4, 16, 2)
TREE_SCAN(4, 32, 2)
TREE_SCAN(4, 64, 2)
TREE_SCAN(8, 8, 1)
TREE_SCAN(8, 16, 1)

#include "fused_scan.cuh"

FUSED_SCAN_MAPS(2, 16, 16, 2, , false)
FUSED_SCAN_MAPS(2, 16, 32, 2, , false)
FUSED_SCAN_MAPS(2, 16, 48, 2, , false)
FUSED_SCAN_MAPS(2, 16, 64, 1, , false)
FUSED_SCAN_MAPS(2, 16, 48, 2, _timed, true)
