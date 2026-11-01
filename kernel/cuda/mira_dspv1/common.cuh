#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define ALPHA_TILE 16


template <typename IO>
__device__ __forceinline__ float load_value(const IO *pointer) {
    return static_cast<float>(*pointer);
}

template <>
__device__ __forceinline__ float load_value<__half>(const __half *pointer) {
    return __half2float(*pointer);
}

template <>
__device__ __forceinline__ float load_value<__nv_bfloat16>(
    const __nv_bfloat16 *pointer) {
    return __bfloat162float(*pointer);
}

template <typename IO>
__device__ __forceinline__ void store_value(IO *pointer, float value) {
    *pointer = static_cast<IO>(value);
}

template <int N>
__device__ __forceinline__ void load_state(const float *pointer, float *state) {
#pragma unroll
    for (int level = 0; level < N; ++level) state[level] = pointer[level];
}

template <int N>
__device__ __forceinline__ void store_state(float *pointer, const float *state) {
#pragma unroll
    for (int level = 0; level < N; ++level) pointer[level] = state[level];
}

template <>
__device__ __forceinline__ void load_state<4>(const float *pointer, float *state) {
    const float4 value = *reinterpret_cast<const float4 *>(pointer);
    state[0] = value.x;
    state[1] = value.y;
    state[2] = value.z;
    state[3] = value.w;
}

template <>
__device__ __forceinline__ void store_state<4>(float *pointer, const float *state) {
    *reinterpret_cast<float4 *>(pointer) = make_float4(
        state[0], state[1], state[2], state[3]);
}

template <>
__device__ __forceinline__ void load_state<2>(const float *pointer, float *state) {
    const float2 value = *reinterpret_cast<const float2 *>(pointer);
    state[0] = value.x;
    state[1] = value.y;
}

template <>
__device__ __forceinline__ void store_state<2>(float *pointer, const float *state) {
    *reinterpret_cast<float2 *>(pointer) = make_float2(state[0], state[1]);
}

template <>
__device__ __forceinline__ void load_state<8>(const float *pointer, float *state) {
    const float4 low = *reinterpret_cast<const float4 *>(pointer);
    const float4 high = *reinterpret_cast<const float4 *>(pointer + 4);
    state[0] = low.x;  state[1] = low.y;  state[2] = low.z;  state[3] = low.w;
    state[4] = high.x; state[5] = high.y; state[6] = high.z; state[7] = high.w;
}

template <>
__device__ __forceinline__ void store_state<8>(float *pointer, const float *state) {
    *reinterpret_cast<float4 *>(pointer) = make_float4(
        state[0], state[1], state[2], state[3]);
    *reinterpret_cast<float4 *>(pointer + 4) = make_float4(
        state[4], state[5], state[6], state[7]);
}

template <>
__device__ __forceinline__ void store_value<__half>(__half *pointer, float value) {
    *pointer = __float2half_rn(value);
}

template <>
__device__ __forceinline__ void store_value<__nv_bfloat16>(
    __nv_bfloat16 *pointer, float value) {
    *pointer = __float2bfloat16_rn(value);
}

/* memory_read is [B, T, K+1, M]: the read direction is per stage, so tap k contracts nu^(k) over
 * the memory axis rather than one direction shared by every tap. */
template <typename IO, int N, int M>
__device__ __forceinline__ float load_read(
    const IO *__restrict__ memory_read, long long row, int row_stride,
    int stage, int memory) {
    return load_value(memory_read + row * row_stride + stage * M + memory);
}

__device__ __forceinline__ float sigmoid_fast(float value) {
    return 1.0f / (1.0f + __expf(-value));
}

__device__ __forceinline__ float silu_fast(float value) {
    return value * sigmoid_fast(value);
}

__device__ __forceinline__ float silu_grad(float value) {
    const float sigmoid = sigmoid_fast(value);
    return sigmoid * (1.0f + value * (1.0f - sigmoid));
}


/* THE COEFFICIENT IS THE PROJECTION, AND NOTHING IS APPLIED TO IT. It enters the rotation through
 * cos = rsqrt(1 + a^2) and sin = a cos, which are bounded for every real a, so there is nothing
 * for an activation to clamp; beta enters through r = 1 / (1 + beta^2) and is bounded likewise.
 * silu_cap, tanh and nsilu were clamps on a reflection coefficient this stage does not have, and
 * tau was tanh's temperature. Both were carried through every signature and discarded here. */
__device__ __forceinline__ float coefficient(float value) {
    return value;
}

__device__ __forceinline__ float coefficient_grad() {
    return 1.0f;
}

template <int L>
__device__ __forceinline__ float warp_broadcast(float value) {
    return __shfl_sync(__activemask(), value, 0, L);
}
