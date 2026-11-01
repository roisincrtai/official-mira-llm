// rwkv_v6.cu -- OUR RWKV-6 WKV kernel, so the RWKV baseline needs no external package.
//
//     S_t[i][j] = w_t[j] S_{t-1}[i][j] + k_t[j] v_t[i]        per head, N x N
//     y_t[i]    = sum_j r_t[j] ( u[j] k_t[j] v_t[i] + S_{t-1}[i][j] )
//
// w arrives as a LOG decay and is exponentiated here; u is RWKV-6's bonus term, which applies
// the CURRENT step's outer product before it enters the state.
//
// STRUCTURE TAKEN FROM THE AUTHORS' OWN KERNEL, RWKV-CUDA-main/wkv6/cuda/wkv6_cuda_v1a.cu, on
// 2026-08-15, after the first honest profile put ours ~80x off published figures. THE MATHS DID
// NOT CHANGE -- it is the same recurrence this file always computed and the same one the numpy
// reference checks. THREE THINGS CHANGED, and the first is the whole gap:
//
//   1. N IS A COMPILE-TIME CONSTANT. It was a kernel ARGUMENT, so `float S[MAX_N]` was indexed
//      by a runtime loop bound; nvcc cannot keep a dynamically indexed array in registers and
//      spilled the entire state row to LOCAL MEMORY, which is backed by HBM. Every step of
//      every thread then read and wrote the state through the memory hierarchy -- destroying
//      the one property the decomposition exists for. As a template parameter with
//      `#pragma unroll`, the row lives in registers. The authors template on it for this reason.
//   2. r, k AND w ARE STAGED IN SHARED MEMORY. Every thread in the block reads all N of them
//      each step, so loading from global inside the inner loop is N^2 loads per (b,h,t) where
//      N would do. One cooperative load, two __syncthreads(), then everyone reads SRAM.
//   3. __launch_bounds__(N, 2), so the compiler budgets registers for two blocks per SM.
//
// THE PARALLEL DECOMPOSITION IS UNCHANGED and matches the authors': one BLOCK per (batch, head),
// one THREAD per output channel i, each thread holding row i of the state. The block is the unit
// of the algorithm -- its threads cooperate through shared memory -- which is why the wrapper
// launches with an explicit grid and block rather than through the generic thread-count path.
//
// Layout, matching fla.ops.rwkv6.chunk_rwkv6's head_first=False so profiler/baselines needs no
// reshaping:  r, k, v, w (B, T, H, N) with w the LOG decay;  u (H, N);  y (B, T, H, N).
// All contiguous float32. This is the authors' layout too: their flat index b*T*C + t*C + h*N+i
// with C = H*N is the same address as ours.
//
// ONE ENTRY POINT PER HEAD SIZE, because N must be known at compile time. The wrapper picks the
// name -- rwkv6_forward_n72 for a head dim of 72 -- and a head size not in this list is refused
// and REPORTED rather than silently truncated. 72 = 576/8 is the width matched_width picks for
// RWKV-6 against MIRA, and its absence from the old 64-wide register row is what made this
// baseline run on PyTorch for three consecutive profiles while every table said "fused".
//
// VERIFIED IN NUMPY: the forward against the reference implementation above, and all five
// gradients against central finite differences at three shapes -- worst 3.5e-09. Re-verified by
// transcription after this rewrite, at every head size below.

// ---------------------------------------------------------------------------------------------
template <int N>
__device__ __forceinline__
void v6_forward(const float* __restrict__ r, const float* __restrict__ k,
                const float* __restrict__ v, const float* __restrict__ lw,
                const float* __restrict__ u, float* __restrict__ y,
                float* __restrict__ ss, const int T, const int H) {
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    __shared__ float sr[N], sk[N], sw[N], su[N];
    float state[N];
#pragma unroll
    for (int j = 0; j < N; ++j) state[j] = 0.0f;

    __syncthreads();
    su[i] = u[h * N + i];                      // the bonus is constant over t: staged once
    __syncthreads();

    for (int t = 0; t < T; ++t) {
        const long long o = (((long long)b * T + t) * H + h) * N;
        __syncthreads();
        sr[i] = r[o + i];
        sk[i] = k[o + i];
        sw[i] = __expf(lw[o + i]);
        __syncthreads();

        const float vi = v[o + i];
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < N; ++j) {
            const float x = sk[j] * vi;                        // the outer product entry [i][j]
            acc = fmaf(sr[j], fmaf(su[j], x, state[j]), acc);
            if (ss != nullptr)
                ss[o * N + (long long)i * N + j] = state[j];   // S_{t-1}, for the backward
            state[j] = fmaf(state[j], sw[j], x);
        }
        y[o + i] = acc;
    }
}

// BACKWARD. With P_t[i][j] = dL/dS_t[i][j], which arrives from y_{t+1} and from S_{t+1}:
//
//     P_{t-1} = gy_t[i] r_t[j] + w_t[j] P_t
//     gr_t[j] = sum_i gy_t[i] ( u[j] k_t[j] v_t[i] + S_{t-1}[i][j] )
//     gu[j]  += sum_i gy_t[i] r_t[j] k_t[j] v_t[i]
//     gk_t[j] = sum_i gy_t[i] r_t[j] u[j] v_t[i]  +  sum_i P_t[i][j] v_t[i]
//     gv_t[i] = gy_t[i] sum_j r_t[j] u[j] k_t[j]  +  sum_j P_t[i][j] k_t[j]
//     glogw_t[j] = ( sum_i P_t[i][j] S_{t-1}[i][j] ) w_t[j]
//
// THREADED BY i AS THE FORWARD IS, so P[i][.] stays in registers; every gradient that sums over
// i (gr, gu, gk, glogw) is finished with atomicAdd, which is the only nondeterminism here.
//
// STILL READS THE SAVED STATE, unlike the authors, who recompute it in three extra passes and
// so never write it at all. That trade is theirs to prefer and ours to postpone: the profile
// times the FORWARD, this backward is correct as written and checked against finite
// differences, and rewriting a verified backward to save memory nobody is short of would be
// work for its own sake. The forward does not write the state at all under no_grad.
template <int N>
__device__ __forceinline__
void v6_backward(const float* __restrict__ r, const float* __restrict__ k,
                 const float* __restrict__ v, const float* __restrict__ lw,
                 const float* __restrict__ u, const float* __restrict__ ss,
                 const float* __restrict__ gy, float* __restrict__ gr,
                 float* __restrict__ gk, float* __restrict__ gv, float* __restrict__ glw,
                 float* __restrict__ gu, const int T, const int H) {
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    __shared__ float sr[N], sk[N], sw[N], su[N];
    float P[N], accu[N];
#pragma unroll
    for (int j = 0; j < N; ++j) { P[j] = 0.0f; accu[j] = 0.0f; }

    __syncthreads();
    su[i] = u[h * N + i];
    __syncthreads();

    for (int t = T - 1; t >= 0; --t) {
        const long long o = (((long long)b * T + t) * H + h) * N;
        const long long sb = o * N + (long long)i * N;
        __syncthreads();
        sr[i] = r[o + i];
        sk[i] = k[o + i];
        sw[i] = __expf(lw[o + i]);
        __syncthreads();

        const float g = gy[o + i];
        const float vi = v[o + i];
        // gv HAS TWO PARTS AND ONLY ONE IS SCALED BY gy_t[i]: the bonus route goes through
        // y_t, the state route does not. They are accumulated separately and combined once.
        float a_scaled = 0.0f, c_plain = 0.0f;
#pragma unroll
        for (int j = 0; j < N; ++j) {
            const float kj = sk[j], rj = sr[j], uj = su[j], w = sw[j];
            const float Sp = ss[sb + j];                       // S_{t-1}[i][j]
            const float Pj = P[j];                             // dL/dS_t, BEFORE advancing

            // through y_t = sum_j r_j ( u_j k_j v_i + S_{t-1}[i][j] )
            atomicAdd(&gr[o + j], g * fmaf(uj, kj * vi, Sp));
            accu[j]   = fmaf(g * rj, kj * vi, accu[j]);        // gu, reduced at the end
            atomicAdd(&gk[o + j], g * rj * uj * vi);
            a_scaled  = fmaf(rj * uj, kj, a_scaled);

            // through S_t = w_t S_{t-1} + k_t v_t
            atomicAdd(&glw[o + j], Pj * Sp * w);               // d/dlog w = d/dw * w
            atomicAdd(&gk[o + j], Pj * vi);
            c_plain   = fmaf(Pj, kj, c_plain);

            P[j] = fmaf(w, Pj, g * rj);                        // P_{t-1}, advanced LAST
        }
        gv[o + i] = fmaf(g, a_scaled, c_plain);
    }
#pragma unroll
    for (int j = 0; j < N; ++j) atomicAdd(&gu[(long long)h * N + j], accu[j]);
}

// ---------------------------------------------------------------------------------------------
// THE ENTRY POINTS. One pair per head size; `extern "C"` so the driver can find them by name in
// the cubin. B and N are passed but unused -- B is implicit in the grid, N is the template
// argument -- and kept so every family under kernel/cuda takes the same (B, T, H, N) tail.
#define R6_ENTRY(NN)                                                                          \
extern "C" __global__ __launch_bounds__(NN, 2)                                                \
void rwkv6_forward_n##NN(const float* __restrict__ r, const float* __restrict__ k,            \
                         const float* __restrict__ v, const float* __restrict__ lw,           \
                         const float* __restrict__ u, float* __restrict__ y,                  \
                         float* __restrict__ ss,                                              \
                         const int B, const int T, const int H, const int N) {                \
    (void)B; (void)N;                                                                         \
    v6_forward<NN>(r, k, v, lw, u, y, ss, T, H);                                              \
}                                                                                             \
extern "C" __global__ __launch_bounds__(NN, 2)                                                \
void rwkv6_backward_n##NN(const float* __restrict__ r, const float* __restrict__ k,           \
                          const float* __restrict__ v, const float* __restrict__ lw,          \
                          const float* __restrict__ u, const float* __restrict__ ss,          \
                          const float* __restrict__ gy, float* __restrict__ gr,               \
                          float* __restrict__ gk, float* __restrict__ gv,                     \
                          float* __restrict__ glw, float* __restrict__ gu,                    \
                          const int B, const int T, const int H, const int N) {               \
    (void)B; (void)N;                                                                         \
    v6_backward<NN>(r, k, v, lw, u, ss, gy, gr, gk, gv, glw, gu, T, H);                       \
}

R6_ENTRY(16)
R6_ENTRY(32)
R6_ENTRY(48)
R6_ENTRY(60)  // 480/8 -- RWKV-7 at a step-16 parameter match
R6_ENTRY(64)  // 512/8
R6_ENTRY(68)  // 544/8 -- a step-16 match for a 544-wide baseline
R6_ENTRY(70)  // 560/8 -- RWKV-6 at a step-16 parameter match
R6_ENTRY(72)  // 576/8
R6_ENTRY(80)
R6_ENTRY(96)
R6_ENTRY(128)
