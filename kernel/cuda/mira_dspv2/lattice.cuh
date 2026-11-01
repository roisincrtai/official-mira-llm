#pragma once
#include "common.cuh"

/* alpha, beta, rho, gamma, x and grad_output are identical for every memory lane of a channel.
 * Broadcasting them from one lane with __shfl_sync cost 2N + N + 3 quarter-rate instructions per
 * channel-token, ahead of the ladder on its dependency chain.  Every lane now issues the same
 * loads instead: the addresses are identical across the lane group, so the L1 serves one request
 * and broadcasts it, the global traffic is unchanged, and the only duplicated work is arithmetic
 * on values already in registers.  The memory write/read coordinates remain lane-specific. */

/* state[k] holds V_{t-1}^(k) on entry and V_t^(k) on exit, for k = 0..N-1; levels[k] is V_t^(k)
 * for k = 0..N.  The input is U_t^(K), the traversal runs k = K..1, and the closure V_t^(0) =
 * U_t^(0) is the last statement.  stage = k - 1, so alpha[stage] is alpha_t^(k). */

/* mode 0 is the lattice stage
 *
 *     U_t^(k-1) = U_t^(k) - alpha_t^(k) V_{t-1}^(k-1)
 *     V_t^(k)   = V_{t-1}^(k-1) + alpha_t^(k) U_t^(k-1)      the updated U
 *
 * and mode 1 is the rotation stage
 *
 *     U_t^(k-1) = cos(alpha_t^(k)) U_t^(k) - sin(alpha_t^(k)) V_{t-1}^(k-1)
 *     V_t^(k)   = sin(alpha_t^(k)) U_t^(k) + cos(alpha_t^(k)) V_{t-1}^(k-1)   the incoming U
 *
 * In mode 1 the staged coefficient is the pair (cos, sin): `slot[stage]` is the cosine and
 * `slot[N + stage]` the sine of the same alpha_t^(k) mode 0 would have used. */

/* mode 2 is the local rotation stage
 *
 *     U_t^(k)   = cos(alpha_t^(k)) U_t^(k-1) - sin(alpha_t^(k)) V_{t-1}^(k)
 *     V_t^(k)   = sin(alpha_t^(k)) U_t^(k-1) + cos(alpha_t^(k)) V_{t-1}^(k)
 *
 * Every lattice stage reads its OWN delayed level rather than the one below it, so both indices
 * ascend together and the traversal runs k = 1..K from the input U_t^(0) = x-tilde_t s_t^T.  The
 * persistent state is V^(1..K), so `state[stage]` holds V_{t-1}^(stage+1) on entry and
 * V_t^(stage+1) on exit and is written in place at its own index; U_t^(K) leaves the ladder
 * unread.  V_t^(0) = U_t^(0), so levels[0] is the current write x-tilde_t s_t^T -- the same outer
 * product that seeds the chain -- and the readout sums k = 0..K.  It stages the same (cos, sin)
 * pair mode 1 does. */

/* mode 3 is the decoupled shifting stage
 *
 *     U_t^(k-1) = r_t^(k) ( cos(theta_t^(k)) U_t^(k) - sin(theta_t^(k)) V_{t-1}^(k-1) )
 *     V_t^(k)   = r_t^(k) ( sin(theta_t^(k)) U_t^(k) + cos(theta_t^(k)) V_{t-1}^(k-1) )
 *
 * with the traversal, the closure V_t^(0) = U_t^(0) and the persistent state V^(0..K-1) of mode 1.
 * The lattice stage turns by theta and is damped by r, and the two come from two projections:
 *
 *     sin(theta) = (1 - alpha^2) / (1 + alpha^2),
 *     cos(theta) = 2 alpha / (1 + alpha^2),  r = 1 / (1 + beta^2)
 *
 * so there is no trigonometric operation or arctangent, only rational arithmetic. It stages
 * three values per coefficient: `slot[stage]` is the cosine, `slot[N + stage]` the sine and
 * `slot[2 * N + stage]` the radius, so its alpha tile is three times mode 0's. */

/* modes 6 to 19 are mode 3's stage under the other cayley_parameterizations, the same radius map
 * from beta and another rotation map from alpha:
 *
 *     mode 3 cauchy_x_mobius         r = 1 / (1 + beta^2),  mode 3's sin and cos
 *     mode 6 cauchy_x_hyperbolic     r = 1 / (1 + beta^2),  sin = tanh(alpha), cos = sech(alpha)
 *     mode 7 cauchy_x_identity       r = 1 / (1 + beta^2),  theta = alpha: sin(alpha), cos(alpha)
 *     mode 8 gaussian_x_mobius       r = exp(-beta^2),      mode 3's sin and cos
 *     mode 9 gaussian_x_hyperbolic   r = exp(-beta^2),      sin = tanh(alpha), cos = sech(alpha)
 *     mode 10 gaussian_x_identity    r = exp(-beta^2),      theta = alpha: sin(alpha), cos(alpha)
 *     mode 11 expexp_x_mobius        r = exp(-exp(beta)),   mode 3's sin and cos
 *     mode 12 expexp_x_hyperbolic    r = exp(-exp(beta)),   sin = tanh(alpha), cos = sech(alpha)
 *     mode 13 expexp_x_identity      r = exp(-exp(beta)),   theta = alpha: sin(alpha), cos(alpha)
 *     mode 14 expsoftplus_x_mobius     r = exp(-softplus(beta)), mode 3's sin and cos
 *     mode 15 expsoftplus_x_hyperbolic r = exp(-softplus(beta)), sin = tanh(alpha), cos = sech(alpha)
 *     mode 16 expsoftplus_x_identity   r = exp(-softplus(beta)), theta = alpha: sin(alpha), cos(alpha)
 *     mode 17 hill_x_mobius          r = beta^2 / (1 + beta^2),  mode 3's sin and cos
 *     mode 18 hill_x_hyperbolic      r = beta^2 / (1 + beta^2),  sin = tanh(alpha), cos = sech(alpha)
 *     mode 19 hill_x_identity        r = beta^2 / (1 + beta^2),  theta = alpha: sin(alpha), cos(alpha)
 *
 * and everything downstream of the staged values -- the ladder, the scan, the readout and the
 * adjoint -- is mode 3's. None is another instantiation: the entry points run the MODE = 3
 * bodies with the runtime value `stage_map = mode - 3`, which is read where the coefficients
 * are staged and where grad_beta is stored. It is the same for every thread of a launch, so the
 * branch is uniform. The hyperbolic rotation is stage_map 3 and the identity rotation stage_map 4;
 * stage_map 5, 6 and 7 are the mobius, hyperbolic and identity rotations with the gaussian radius,
 * stage_map 8, 9 and 10 with the expexp radius, stage_map 11, 12 and 13 with the expsoftplus
 * radius r = exp(-softplus(beta)), and stage_map 14, 15 and 16 with the hill radius
 * r = beta^2 / (1 + beta^2). */

/* THE STAGE IS A TEMPLATE PARAMETER, NOT A RUNTIME ARGUMENT.  Every helper and every phase body
 * below carries `int MODE` alongside its existing parameters and is instantiated once per mode,
 * so each `MODE ==` test is a constant the compiler folds before the lattice stage loops are unrolled:
 * no comparison, no branch and no live value for the stage survives inside a loop.  The runtime
 * `mode` scalar is read exactly once, in the extern "C" entry point, where the branch is uniform
 * across the block and outside every loop.  The entry points and their names are unchanged and
 * every argument list keeps its order, pointers then ints; only the body is instantiated four
 * times.  raw_beta is read under modes 3 and 4 alone and may be null under modes 0, 1 and 2. */

template <int N, int MODE>
__device__ __forceinline__ void advance(
    float input, const float *alpha, float *state, float *levels = nullptr) {
    float U = input;
    if (MODE == 2) {
#pragma unroll
        for (int stage = 0; stage < N; ++stage) {
            const float delayed = state[stage];
            const float cosine = alpha[stage];
            const float sine = alpha[N + stage];
            const float level = fmaf(sine, U, cosine * delayed);
            U = fmaf(cosine, U, -(sine * delayed));
            state[stage] = level;
            if (levels) levels[stage + 1] = level;
        }
        if (levels) levels[0] = input;
    } else {
#pragma unroll
        for (int stage = N - 1; stage >= 0; --stage) {
            const float delayed = state[stage];
            float level;
            if (MODE == 3) {
                const float scaled_cosine = alpha[stage];
                const float scaled_sine = alpha[N + stage];
                level = fmaf(scaled_cosine, delayed, scaled_sine * U);
                U = fmaf(scaled_cosine, U, -(scaled_sine * delayed));
            } else if (MODE) {
                const float cosine = alpha[stage];
                const float sine = alpha[N + stage];
                level = fmaf(cosine, delayed, sine * U);
                U = fmaf(cosine, U, -(sine * delayed));
            } else {
                U = fmaf(-alpha[stage], delayed, U);
                level = fmaf(alpha[stage], U, delayed);
            }
            if (stage + 1 < N) state[stage + 1] = level;
            if (levels) levels[stage + 1] = level;
        }
        state[0] = U;
        if (levels) levels[0] = U;
    }
}

template <int N, int V, int MODE>
__device__ __forceinline__ void advance_vectors(
    const float *input, const float alpha[V][N], const float *slot,
    float state[V][N], float (*levels)[N + 1] = nullptr) {
    float U[V];
#pragma unroll
    for (int vector = 0; vector < V; ++vector) U[vector] = input[vector];
    if (MODE == 2) {
#pragma unroll
        for (int stage = 0; stage < N; ++stage) {
            const float cosine = slot[stage];
            const float sine = slot[N + stage];
#pragma unroll
            for (int vector = 0; vector < V; ++vector) {
                const float delayed = state[vector][stage];
                const float level = fmaf(sine, U[vector], cosine * delayed);
                U[vector] = fmaf(cosine, U[vector], -(sine * delayed));
                state[vector][stage] = level;
                if (levels) levels[vector][stage + 1] = level;
            }
        }
        if (levels) {
#pragma unroll
            for (int vector = 0; vector < V; ++vector) levels[vector][0] = input[vector];
        }
    } else {
#pragma unroll
        for (int stage = N - 1; stage >= 0; --stage) {
            if (MODE == 3) {
                const float scaled_cosine = slot[stage];
                const float scaled_sine = slot[N + stage];
#pragma unroll
                for (int vector = 0; vector < V; ++vector) {
                    const float delayed = state[vector][stage];
                    const float level = fmaf(
                        scaled_cosine, delayed, scaled_sine * U[vector]);
                    U[vector] = fmaf(
                        scaled_cosine, U[vector], -(scaled_sine * delayed));
                    if (stage + 1 < N) state[vector][stage + 1] = level;
                    if (levels) levels[vector][stage + 1] = level;
                }
            } else if (MODE) {
                const float cosine = slot[stage];
                const float sine = slot[N + stage];
#pragma unroll
                for (int vector = 0; vector < V; ++vector) {
                    const float delayed = state[vector][stage];
                    const float level = fmaf(cosine, delayed, sine * U[vector]);
                    U[vector] = fmaf(cosine, U[vector], -(sine * delayed));
                    if (stage + 1 < N) state[vector][stage + 1] = level;
                    if (levels) levels[vector][stage + 1] = level;
                }
            } else {
#pragma unroll
                for (int vector = 0; vector < V; ++vector) {
                    const float delayed = state[vector][stage];
                    U[vector] = fmaf(-alpha[vector][stage], delayed, U[vector]);
                    const float level = fmaf(alpha[vector][stage], U[vector], delayed);
                    if (stage + 1 < N) state[vector][stage + 1] = level;
                    if (levels) levels[vector][stage + 1] = level;
                }
            }
        }
#pragma unroll
        for (int vector = 0; vector < V; ++vector) {
            state[vector][0] = U[vector];
            if (levels) levels[vector][0] = U[vector];
        }
    }
}

template <int N, int MODE>
__device__ __forceinline__ void reverse_advance(
    const float *alpha, const float *local_level_gradient, float *future) {
    float level_gradient[N + 1];
#pragma unroll
    for (int stage = 0; stage <= N; ++stage)
        level_gradient[stage] = local_level_gradient[stage];
    /* mode 2 carries V^(1..K), so `future[stage]` lands on level_gradient[stage + 1]; the chain
     * descends from grad U_t^(K) = 0 because that wave leaves the ladder unread. */
    if (MODE == 2) {
#pragma unroll
        for (int stage = 0; stage < N; ++stage)
            level_gradient[stage + 1] += future[stage];
        float grad_u = 0.0f;
#pragma unroll
        for (int stage = N - 1; stage >= 0; --stage) {
            const float grad_v = level_gradient[stage + 1];
            const float cosine = alpha[stage];
            const float sine = alpha[N + stage];
            future[stage] = fmaf(cosine, grad_v, -(sine * grad_u));
            grad_u = fmaf(cosine, grad_u, sine * grad_v);
        }
    } else {
#pragma unroll
        for (int stage = 0; stage < N; ++stage)
            level_gradient[stage] += future[stage];
        float grad_u = level_gradient[0];
#pragma unroll
        for (int stage = 0; stage < N; ++stage) {
            const float grad_v = level_gradient[stage + 1];
            if (MODE == 3) {
                const float scaled_cosine = alpha[stage];
                const float scaled_sine = alpha[N + stage];
                future[stage] = fmaf(
                    scaled_cosine, grad_v, -(scaled_sine * grad_u));
                grad_u = fmaf(scaled_cosine, grad_u, scaled_sine * grad_v);
            } else if (MODE) {
                const float cosine = alpha[stage];
                const float sine = alpha[N + stage];
                future[stage] = fmaf(cosine, grad_v, -(sine * grad_u));
                grad_u = fmaf(cosine, grad_u, sine * grad_v);
            } else {
                grad_u = fmaf(alpha[stage], grad_v, grad_u);
                future[stage] = fmaf(-alpha[stage], grad_u, grad_v);
            }
        }
    }
}

template <int N, int V, int MODE>
__device__ __forceinline__ void reverse_advance_vectors(
    const float alpha[V][N], const float *slot, const float local[V][N + 1],
    float future[V][N]) {
    float level_gradient[V][N + 1];
    float grad_u[V];
#pragma unroll
    for (int stage = 0; stage <= N; ++stage) {
#pragma unroll
        for (int vector = 0; vector < V; ++vector)
            level_gradient[vector][stage] = local[vector][stage];
    }
    if (MODE == 2) {
#pragma unroll
        for (int stage = 0; stage < N; ++stage) {
#pragma unroll
            for (int vector = 0; vector < V; ++vector)
                level_gradient[vector][stage + 1] += future[vector][stage];
        }
#pragma unroll
        for (int vector = 0; vector < V; ++vector) grad_u[vector] = 0.0f;
#pragma unroll
        for (int stage = N - 1; stage >= 0; --stage) {
            const float cosine = slot[stage];
            const float sine = slot[N + stage];
#pragma unroll
            for (int vector = 0; vector < V; ++vector) {
                const float grad_v = level_gradient[vector][stage + 1];
                future[vector][stage] = fmaf(cosine, grad_v, -(sine * grad_u[vector]));
                grad_u[vector] = fmaf(cosine, grad_u[vector], sine * grad_v);
            }
        }
    } else {
#pragma unroll
        for (int stage = 0; stage < N; ++stage) {
#pragma unroll
            for (int vector = 0; vector < V; ++vector)
                level_gradient[vector][stage] += future[vector][stage];
        }
#pragma unroll
        for (int vector = 0; vector < V; ++vector) grad_u[vector] = level_gradient[vector][0];
#pragma unroll
        for (int stage = 0; stage < N; ++stage) {
            if (MODE == 3) {
                const float scaled_cosine = slot[stage];
                const float scaled_sine = slot[N + stage];
#pragma unroll
                for (int vector = 0; vector < V; ++vector) {
                    const float grad_v = level_gradient[vector][stage + 1];
                    future[vector][stage] = fmaf(
                        scaled_cosine, grad_v, -(scaled_sine * grad_u[vector]));
                    grad_u[vector] = fmaf(
                        scaled_cosine, grad_u[vector], scaled_sine * grad_v);
                }
            } else if (MODE) {
                const float cosine = slot[stage];
                const float sine = slot[N + stage];
#pragma unroll
                for (int vector = 0; vector < V; ++vector) {
                    const float grad_v = level_gradient[vector][stage + 1];
                    future[vector][stage] = fmaf(cosine, grad_v, -(sine * grad_u[vector]));
                    grad_u[vector] = fmaf(cosine, grad_u[vector], sine * grad_v);
                }
            } else {
#pragma unroll
                for (int vector = 0; vector < V; ++vector) {
                    const float grad_v = level_gradient[vector][stage + 1];
                    grad_u[vector] = fmaf(alpha[vector][stage], grad_v, grad_u[vector]);
                    future[vector][stage] = fmaf(
                        -alpha[vector][stage], grad_u[vector], grad_v);
                }
            }
        }
    }
}

/* alpha_t^(k) is identical for every lane of a channel, so activating it per lane costs
 * M/L * N transcendentals per warp-token where N distinct values exist.  The
 * tile below activates each value once for a run of ALPHA_TILE tokens and the token loop reads
 * it back from shared memory. */
/* THE IDENTITY ROTATION, theta = alpha: sin(alpha) and cos(alpha). The cubin is built with
 * --use_fast_math, so sincosf is the __sincosf intrinsic, whose absolute error is about 2^-21.4
 * on [-pi, pi] and grows with |alpha| outside it. alpha is not bounded, so it is first reduced to
 * [-pi, pi] by a two-constant (Cody-Waite) subtraction of the nearest multiple of 2 pi, which
 * keeps the reduced argument accurate to float32 rounding for every |alpha| < 2^17. */
__device__ __forceinline__ void identity_sincos(float value, float *sine, float *cosine) {
    const float turns = rintf(value * 0.15915494309189535f);
    float reduced = fmaf(-turns, 6.28318548202514648f, value);
    reduced = fmaf(-turns, -1.74845553e-07f, reduced);
    __sincosf(reduced, sine, cosine);
}

/* THE RADIUS MAP OF A stage_map, AND WHICH ROTATION IT CARRIES. stage_map 0, 3, 4 are the
 * cauchy radius r = 1 / (1 + beta^2) with the mobius, hyperbolic and identity rotations; 5, 6, 7
 * are the gaussian radius r = exp(-beta^2), 8, 9, 10 the expexp radius r = exp(-exp(beta)),
 * 11, 12, 13 the expsoftplus radius r = exp(-softplus(beta)) and 14, 15, 16 the hill radius
 * r = beta^2 / (1 + beta^2), each with the same three rotations. radius_of returns 0 (cauchy),
 * 1 (gaussian), 2 (expexp), 3 (expsoftplus) or 4 (hill); the rotation is returned in the cauchy
 * numbering, 0 (mobius), 3 (hyperbolic) or 4 (identity). */
__device__ __forceinline__ int radius_of(int stage_map) {
    if (stage_map >= 14) return 4;
    return stage_map >= 11 ? 3 : (stage_map >= 8 ? 2 : (stage_map >= 5 ? 1 : 0));
}

__device__ __forceinline__ int rotation_of(int stage_map) {
    const int base = stage_map >= 14 ? stage_map - 14
        : (stage_map >= 11 ? stage_map - 11
        : (stage_map >= 8 ? stage_map - 8 : (stage_map >= 5 ? stage_map - 5 : -1)));
    return base < 0 ? stage_map : (base == 0 ? 0 : base + 2);
}

/* THE RADIUS OF A STAGE AND ITS CACHED RATE. The backward needs (dr / dbeta) / r, cached as
 * `rate` and multiplied at the grad_beta store by the stage's beta factor:
 *     cauchy  r = 1 / (1 + beta^2)   (dr/dbeta) / r = -2 beta r   rate 2 r       factor -beta
 *     gaussian r = exp(-beta^2)      (dr/dbeta) / r = -2 beta     rate 2         factor -beta
 *     expexp  r = exp(-exp(beta))    (dr/dbeta) / r = -exp(beta)  rate exp(beta) factor -1
 *     expsoftplus (dr/dbeta) / r = -exp(s_r^(c)) sigmoid(beta), factor -1
 *     hill    r = q / (1 + q), q = beta^2   (dr/dbeta) / r = 2 beta / (q (1 + q))
 *             rate -2 beta (1 - r) / max(q, 1e-8)   factor -1
 * The expexp beta is capped at 80 before the inner exponential, as in the model: above it r and
 * dr/dbeta are already 0 in float32, and exp(beta) would reach inf past 88.7. The expsoftplus
 * radius uses stable softplus(beta), and the per-channel s_r pointer is null for unscaled calls.
 * The hill radius forms 1 - r = 1 / (1 + q) directly, takes r = 1 - (1 - r) above q = 1 and
 * q (1 - r) at and below it, and guards the rate at q = 0 (beta = 0, r = 0) with max(q, 1e-8). */
__device__ __forceinline__ float stage_radius(int kind, float beta, float *rate) {
    if (kind == 4) {
        const float square = beta * beta;
        const float complement = 1.0f / (1.0f + square);
        *rate = -2.0f * (beta / fmaxf(square, 1.0e-8f)) * complement;
        return square > 1.0f ? 1.0f - complement : square * complement;
    }
    if (kind == 3) {
        const float tail = __expf(-fabsf(beta));
        const float inverse = 1.0f / (1.0f + tail);
        const float small = tail * inverse;
        *rate = beta >= 0.0f ? inverse : small;
        const float softplus = fmaxf(beta, 0.0f) + log1pf(tail);
        return __expf(-softplus);
    }
    if (kind == 2) {
        const float growth = __expf(fminf(beta, 80.0f));
        *rate = growth;
        return __expf(-growth);
    }
    if (kind == 1) {
        *rate = 2.0f;
        return __expf(-beta * beta);
    }
    const float radius = 1.0f / fmaf(beta, beta, 1.0f);
    *rate = 2.0f * radius;
    return radius;
}

template <typename IO, int N, int G, int MODE, bool DERIVATIVES>
__device__ __forceinline__ void stage_alpha(
    const IO *__restrict__ raw_alpha, const IO *__restrict__ raw_beta,
    float *cache,
    int batch_index, int tokens, int channels, int channel_base,
    int tile_first, int tile_tokens,
    int alpha_row_stride, int beta_row_stride, int stage_map) {
    const int span = MODE == 3 ? (DERIVATIVES ? 4 * N : 2 * N) : (MODE ? 2 * N : N);
    for (int index = threadIdx.x; index < tile_tokens * G * N;
         index += blockDim.x) {
        const int stage = index % N;
        const int rest = index / N;
        const int local = rest % G;
        const int local_token = rest / G;
        const int channel = channel_base + local;
        float value = 0.0f;
        float damping = 0.0f;
        if (channel < channels) {
            const long long row =
                static_cast<long long>(batch_index) * tokens + tile_first + local_token;
            value = coefficient(
                load_value(raw_alpha
                           + row * alpha_row_stride + channel * N + stage));
            if (MODE == 3)
                damping = load_value(
                    raw_beta + row * beta_row_stride + channel * N + stage);
        }
        float *entry = cache + (local_token * G + local) * span + stage;
        const int radius_kind = radius_of(stage_map);
        const int rotation = rotation_of(stage_map);
        if (MODE == 3 && rotation != 0) {
            float cosine, sine, angle_rate;
            if (rotation == 4) {
                /* THE IDENTITY ROTATION: theta = alpha, so d theta / d alpha is 1. */
                identity_sincos(value, &sine, &cosine);
                angle_rate = 1.0f;
            } else {
                /* THE HYPERBOLIC ROTATION:
                 *     sin = tanh(alpha) = (1 - e^2) / (1 + e^2),  cos = sech(alpha) = 2 e / (1 + e^2),
                 * with e = exp(-|alpha|), so both are one exponential and one reciprocal and
                 * neither overflows. theta = arctan(sinh(alpha)), so d theta / d alpha is
                 * sech(alpha), the cosine. */
                const float decay = __expf(-fabsf(value));
                const float decay_squared = decay * decay;
                const float angle_inverse = 1.0f / (1.0f + decay_squared);
                cosine = 2.0f * decay * angle_inverse;
                sine = copysignf((1.0f - decay_squared) * angle_inverse, value);
                angle_rate = cosine;
            }
            /* the cauchy, gaussian, expexp, expsoftplus or hill radius and its cached rate, see
             * stage_radius. */
            float radius_rate;
            const float radius = stage_radius(radius_kind, damping, &radius_rate);
            entry[0] = radius * cosine;
            entry[N] = radius * sine;
            if (DERIVATIVES) {
                entry[2 * N] = angle_rate;
                entry[3 * N] = radius_rate;
            }
        } else if (MODE == 3 && radius_kind != 0) {
            /* gaussian_x_mobius, expexp_x_mobius and expsoftplus_x_mobius: mode 3's rotation,
             * sin = (1 - alpha^2) / (1 + alpha^2) and cos = 2 alpha / (1 + alpha^2), written in
             * 1 / alpha above |alpha| = 1, times the selected radius. */
            const bool near_zero = fabsf(value) <= 1.0f;
            const float base = near_zero ? value : 1.0f / value;
            const float square = base * base;
            const float angle_denominator = 1.0f + square;
            float radius_rate;
            const float radius = stage_radius(radius_kind, damping, &radius_rate);
            const float scale = radius / angle_denominator;
            entry[0] = 2.0f * base * scale;
            entry[N] = (near_zero ? (1.0f - square) : (square - 1.0f)) * scale;
            if (DERIVATIVES) {
                entry[2 * N] = -2.0f * (near_zero ? 1.0f : square) / angle_denominator;
                entry[3 * N] = radius_rate;
            }
        } else if (MODE == 3) {
            /* THE SHIFTED CAYLEY ROTATION IN CLOSED FORM, WITH NO INVERSE AND NO ITERATION:
             *     sin = (1 - alpha^2) / (1 + alpha^2),   cos = 2 alpha / (1 + alpha^2).
             * Thus theta = pi/2 - 2 arctan(alpha), modulo 2 pi, and R is orthogonal for every
             * real alpha. Above |alpha| = 1 the same rational map is written in 1 / alpha,
             * which is what the reference does, so alpha^2 never overflows. */
            /* ONE RECIPROCAL, NOT TWO. r cos and r sin share the denominator
             * (1 + alpha^2)(1 + beta^2), so the angle scale and the radius come from a single
             * division and r itself is recovered as inverse * (1 + alpha^2). */
            const bool near_zero = fabsf(value) <= 1.0f;
            const float base = near_zero ? value : 1.0f / value;
            const float square = base * base;
            const float angle_denominator = 1.0f + square;
            const float radius_denominator = fmaf(damping, damping, 1.0f);
            const float inverse = 1.0f / (angle_denominator * radius_denominator);
            /* THE RADIUS CARRIES NO SQUARE ROOT: r = 1 / (1 + beta^2), which is in (0, 1] for
             * every real beta, so the stage is Schur stable at every parameter value. */
            const float radius = inverse * angle_denominator;
            entry[0] = 2.0f * base * inverse;
            entry[N] = (near_zero ? (1.0f - square) : (square - 1.0f)) * inverse;
            if (DERIVATIVES) {
                /* d theta / d alpha = -2 q with q = 1 / (1 + alpha^2). The rotation backward
                 * already forms its derivative with respect to theta, so the cached factor is
                 * negative after shifting theta from 2 arctan(alpha) to pi/2 - 2 arctan(alpha).
                 * q is the same inverse again: at |alpha| <= 1 it is inverse (1 + beta^2), and
                 * above it the rational map is written in 1 / alpha, so alpha^2 = 1 / square and
                 * q carries the extra factor square. */
                entry[2 * N] = -2.0f * (near_zero ? 1.0f : square) / angle_denominator;
                /* r = 1 / (1 + beta^2) gives dr/dbeta = -2 beta r^2, so (dr/dbeta) / r is
                 * -2 beta r and the cached factor is 2 r; the -beta is applied at the store. */
                entry[3 * N] = 2.0f * radius;
            }
        } else if (MODE == 2) {
            const float cosine = rsqrtf(fmaf(value, value, 1.0f));
            entry[0] = cosine;
            entry[N] = value * cosine;
        } else if (MODE) {
            float sine, cosine;
            __sincosf(value, &sine, &cosine);
            entry[0] = cosine;
            entry[N] = sine;
        } else {
            entry[0] = value;
        }
    }
}

template <int N, int G>
__device__ __forceinline__ int alpha_slot(
    int local_token, int local_channel, int span) {
    return (local_token * G + local_channel) * span;
}

template <typename IO, int N>
__device__ __forceinline__ void load_coefficients(const IO *__restrict__ source, float *values) {
    if (N == 4 && (reinterpret_cast<unsigned long long>(source) % (4 * sizeof(IO))) == 0) {
        if (sizeof(IO) == 4) {
            const float4 packed = *reinterpret_cast<const float4 *>(source);
            values[0] = packed.x; values[1] = packed.y;
            values[2] = packed.z; values[3] = packed.w;
        } else {
            const uint2 packed = *reinterpret_cast<const uint2 *>(source);
            const IO *elements = reinterpret_cast<const IO *>(&packed);
#pragma unroll
            for (int i = 0; i < 4; ++i) values[i] = load_value(elements + i);
        }
    } else if (N == 2 && sizeof(IO) == 2
               && (reinterpret_cast<unsigned long long>(source) % 4) == 0) {
        /* THE PAIR IS FOUR CONTIGUOUS BYTES AT N == 2.  The base torch gives is sixteen-byte
         * aligned, the channel offset is channel * N = two elements, and the row stride of
         * [B, T, d_e, K] is d_e * K, so every term is a multiple of two elements and the address
         * is four-byte aligned.  The runtime test keeps the access safe for a stride where that
         * does not hold, as the N == 4 case above does. */
        const unsigned int packed = *reinterpret_cast<const unsigned int *>(source);
        const IO *elements = reinterpret_cast<const IO *>(&packed);
        values[0] = load_value(elements);
        values[1] = load_value(elements + 1);
    } else {
#pragma unroll
        for (int i = 0; i < N; ++i) values[i] = load_value(source + i);
    }
}

/* THE COEFFICIENT MATHS, GIVEN THE TWO ROWS.  Split out of load_stage so a caller that already
 * holds the row pointers does not re-derive them: see scan_phase_one, where the token loop walks
 * `row` by one and the two 64-bit `base + row * stride` chains were being rebuilt from scratch
 * every token.  load_stage below still forms them from (row, channel, stride) and calls this, so
 * there is exactly one copy of the arithmetic and every other caller is untouched. */
/* THE STAGE MAP OF ONE (stream, token), FROM ITS LOADED alpha AND beta VALUES. load_stage_at's
 * arithmetic, split out (mira_v5_stable) so the fused scan can call it with a compile-time stage_map and
 * the compiler keeps only that map's instructions; every caller runs this one body. */
template <int N>
__device__ __forceinline__ void stage_slots(
    const float *alpha_values, const float *beta_values, float *slot, int stage_map) {
#pragma unroll
    for (int stage = 0; stage < N; ++stage) {
        const float value = coefficient(
            alpha_values[stage]);
        const float damping = beta_values[stage];
        const int radius_kind = radius_of(stage_map);
        const int rotation = rotation_of(stage_map);
        float unused_rate;
        if (rotation == 4) {
            /* The identity stage of stage_alpha: r cos(alpha) and r sin(alpha), with the radius
             * of stage_radius. */
            float sine, cosine;
            identity_sincos(value, &sine, &cosine);
            const float radius = stage_radius(radius_kind, damping, &unused_rate);
            slot[stage] = radius * cosine;
            slot[N + stage] = radius * sine;
            continue;
        }
        if (rotation == 3) {
            /* The hyperbolic stage of stage_alpha: r sech(alpha) and r tanh(alpha), with
             * r = 1 / (1 + beta^2) or, under the gaussian, expexp and expsoftplus radii,
             * r = exp(-beta^2), exp(-exp(beta)) and exp(-exp(s_r^(c)) softplus(beta)). */
            const float decay = __expf(-fabsf(value));
            const float decay_squared = decay * decay;
            const float scale = radius_kind != 0
                ? stage_radius(radius_kind, damping, &unused_rate) / (1.0f + decay_squared)
                : 1.0f / ((1.0f + decay_squared) * fmaf(damping, damping, 1.0f));
            slot[stage] = 2.0f * decay * scale;
            slot[N + stage] = copysignf((1.0f - decay_squared) * scale, value);
            continue;
        }
        const bool near_zero = fabsf(value) <= 1.0f;
        const float base = near_zero ? value : 1.0f / value;
        const float square = base * base;
        /* ONE RECIPROCAL FOR THE PAIR: r cos and r sin share the denominator
         * (1 + alpha^2)(1 + beta^2); under the gaussian, expexp and expsoftplus radii
         * r = exp(-beta^2), exp(-exp(beta)) and exp(-exp(s_r^(c)) softplus(beta)) multiply. */
        const float inverse = radius_kind != 0
            ? stage_radius(radius_kind, damping, &unused_rate) / (1.0f + square)
            : 1.0f / ((1.0f + square) * fmaf(damping, damping, 1.0f));
        slot[stage] = 2.0f * base * inverse;
        slot[N + stage] = (near_zero ? (1.0f - square) : (square - 1.0f)) * inverse;
    }
}

template <typename IO, int N>
__device__ __forceinline__ void load_stage_at(
    const IO *__restrict__ alpha_row, const IO *__restrict__ beta_row,
    bool active, float *slot, int stage_map) {
    if (!active) {
#pragma unroll
        for (int entry = 0; entry < 2 * N; ++entry) slot[entry] = 0.0f;
        return;
    }
    float alpha_values[N], beta_values[N];
    load_coefficients<IO, N>(alpha_row, alpha_values);
    load_coefficients<IO, N>(beta_row, beta_values);
    stage_slots<N>(alpha_values, beta_values, slot, stage_map);
}

template <typename IO, int N, int L>
__device__ __forceinline__ void load_stage(
    const IO *__restrict__ raw_alpha, const IO *__restrict__ raw_beta,
    long long row, int channel,
    int alpha_row_stride, int beta_row_stride, bool active, float *slot, int stage_map) {
    load_stage_at<IO, N>(raw_alpha + row * alpha_row_stride + channel * N,
                         raw_beta + row * beta_row_stride + channel * N,
                         active, slot, stage_map);
}

template <typename IO, int N, int L>
__device__ __forceinline__ void load_rho(
    const IO *__restrict__ raw_rho, long long offset, bool active, float *rho) {
    if (!active) {
#pragma unroll
        for (int stage = 0; stage <= N; ++stage) rho[stage] = 0.0f;
        return;
    }
    const IO *__restrict__ rho_row = raw_rho + offset;
    if (N == 2 && sizeof(IO) == 2) {
        /* THE TRIPLE IS SIX CONTIGUOUS BYTES AT N == 2, and the channel offset is
         * channel * (N + 1) elements, so consecutive channels alternate between four- and
         * two-byte alignment and no one pair of stages is always aligned.  A two-byte element is
         * two-byte aligned and the base and row stride contribute multiples of four bytes, so the
         * row is zero or two modulo four and exactly one of the pairs (0, 1) and (1, 2) is
         * four-byte aligned: the 32-bit access takes that pair and the remaining stage is read on
         * its own.  Three 16-bit accesses become one 32-bit and one 16-bit either way. */
        const bool aligned = (reinterpret_cast<unsigned long long>(rho_row) % 4) == 0;
        const unsigned int packed = *reinterpret_cast<const unsigned int *>(
            aligned ? rho_row : rho_row + 1);
        const IO *elements = reinterpret_cast<const IO *>(&packed);
        const float low = load_value(elements);
        const float high = load_value(elements + 1);
        const float single = load_value(rho_row + (aligned ? 2 : 0));
        rho[0] = aligned ? low : single;
        rho[1] = aligned ? high : low;
        rho[2] = aligned ? single : high;
    } else {
#pragma unroll
        for (int stage = 0; stage <= N; ++stage)
            rho[stage] = load_value(rho_row + stage);
    }
}
