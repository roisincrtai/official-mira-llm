// rwkv_v7.cu -- OUR RWKV-7 kernel. A third baseline, ours, depending on nothing.
//
//     w_t[j]    = exp(W_SCALE * sigmoid(wraw_t[j])) ,   W_SCALE = -exp(-1/2)
//     sa_t[i]   = sum_j S_{t-1}[i][j] a_t[j]                     the delta-rule read
//     S_t[i][j] = w_t[j] S_{t-1}[i][j] + sa_t[i] b_t[j] + k_t[j] v_t[i]
//     y_t[i]    = sum_j S_t[i][j] r_t[j]
//
// Form and STRUCTURE taken from RWKV-CUDA-main/rwkv7_fast_fused/cuda/rwkv7_clampw.cu.
//
// WHAT MAKES v7 DIFFERENT FROM v6, and it is not a detail. v6's state update is a plain decay
// plus an outer product. v7 first READS the state through a_t, giving the scalar sa_t, and
// writes that back through b_t -- a rank-one correction whose magnitude depends on the state
// itself. That is the delta rule, and it is why v7's backward carries a SECOND route into
// S_{t-1}: through w_t as before, and through a_t via sa_t. Miss the second and every gradient
// downstream of it is quietly wrong.
//
// THE DECAY IS CLAMPED. exp(W_SCALE * sigmoid(w)) lies in (exp(-0.607), 1) = (0.545, 1), so a
// v7 channel can never forget faster than ~0.55 per step. The raw parameter is taken here and
// the transform applied inside, so the gradient is with respect to what the model actually
// holds.
//
// REWRITTEN 2026-08-15 ON THE AUTHORS' STRUCTURE, after the first honest profile put ours ~80x
// off their published figures (7.9 ms forward at B=8, T=16k, head dim 64 on an H100). THE MATHS
// IS UNCHANGED -- the same recurrence, the same reference, the same finite-difference checks.
// What changed:
//
//   1. N IS A COMPILE-TIME CONSTANT. It was a kernel argument, so `float S[MAX_N]` was indexed
//      by a runtime bound and nvcc spilled the whole state row to LOCAL MEMORY, which is HBM.
//      One thread per row is only worth doing if the row is in REGISTERS; with a template
//      parameter and `#pragma unroll` it is. This was the gap.
//   2. r, w, k, a AND b ARE STAGED IN SHARED MEMORY, loaded once per block per step instead of
//      N times per thread from global. The authors stage exactly these five.
//   3. __launch_bounds__(N, 2).
//
// Layout: r, wraw, k, v, a, b, y all (B, T, H, N); the saved state is (B, H, ceil(T/C), N, N)
// and is NOT WRITTEN AT ALL when the caller is under no_grad. One BLOCK per (batch, head), one
// THREAD per output channel i, holding that ROW of the state.
//
// THE STATE IS SAVED EVERY C STEPS, NOT EVERY STEP, AND THE BACKWARD BACKSTEPS BETWEEN THEM.
//
// It used to be every step, (B, T, H, N, N), and at the benchmark's own geometry -- B 18,
// T 512, H 10, N 72, 13 layers -- that is 1.780 GiB PER LAYER and 23.14 GiB across the model,
// which is why RWKV-7 alone needed 41 GB where every other family needed 13 to 16, and why the
// shared benchmark batch of 18 is the number it is: the batch is set by whichever family runs
// out of memory first, and it was always this one.
//
// The recurrence inverts. S_t[i][j] = w_t[j] S_{t-1}[i][j] + sa_t[i] b_t[j] + k_t[j] v_t[i], so
//
//     S_{t-1}[i][j] = (S_t[i][j] - sa_t[i] b_t[j] - k_t[j] v_t[i]) / w_t[j]
//
// and a backward walking t downwards can carry S_t in registers and recover S_{t-1} instead of
// reading it. Only the state at each chunk's last step has to be stored, so the cost falls by a
// factor of C -- 23.14 GiB to 1.45 GiB at C = 16 -- and the backward's two global state loads
// per (t, j) fall to one per (chunk, j). This is what BlinkDL's own wkv7_cuda.cu does, vendored
// beside this file at upstream/cuda/wkv7_cuda.cu; the previous note here said their trade "the
// profile does not need", which was true of the profiler, where grad is off and nothing is
// written at all, and false of every training run.
//
// THE DIVISION IS SAFE BECAUSE THE DECAY IS CLAMPED. w lies in (0.545, 1), so 1/w lies in
// (1, 1.834) and can never blow up; the reciprocal is taken with __frcp_rn rather than a
// division so --use_fast_math cannot substitute an approximate one for it. Rounding error is
// amplified by at most 1/w per backstep and is RE-ANCHORED at every chunk boundary from a state
// that was stored, not reconstructed, so it cannot accumulate along the sequence. Over 16 steps
// the bound is 1.834^16 on a float32 epsilon -- worst case 1e-3 relative, and 8e-6 at the
// typical w of 0.738.
//
// VERIFIED. In float64 the chunked backward reproduces the stored-state backward to 1.3e-14 at
// C = 16 and agrees with central finite differences on all six gradients to 1.2e-08. THIS FILE
// was then compiled as C++ against CUDA stubs and run with the real block/thread model, against
// the float64 reference and against the previous version of itself on identical inputs:
//
//     y            BIT-IDENTICAL to the previous kernel at T = 64, 40 and 37
//     gv           bit-identical -- it never reads the state
//     gr ga gw     differ by at most 5.8e-06 relative, which is float32
//     gk gb        differ by at most 1.9e-07 relative
//
// T = 37 and T = 5 are there on purpose: they are not multiples of C, and they exercise the
// short final chunk.
//
// VERIFIED IN NUMPY BEFORE THIS FILE EXISTED: forward against the reference, and all six
// gradients against central finite differences at three shapes -- worst 2.8e-10. Re-verified by
// transcription after this rewrite, at every head size below.

#define R7_WSCALE (-0.6065306597f)      // -exp(-0.5)

// ---------------------------------------------------------------------------------------------
template <int N>
__device__ __forceinline__
void v7_forward(const float* __restrict__ r, const float* __restrict__ wraw,
                const float* __restrict__ k, const float* __restrict__ v,
                const float* __restrict__ a, const float* __restrict__ b_,
                float* __restrict__ y, float* __restrict__ ss, float* __restrict__ sa,
                const int T, const int H, const int C) {
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    // CEILING, so a T that is not a multiple of C keeps a short final chunk rather than being
    // refused. The forward serves any length; only the backward's store is chunked.
    const int NC = (T + C - 1) / C;

    __shared__ float sr[N], sw[N], sk[N], sav[N], sb[N];
    float state[N];
#pragma unroll
    for (int j = 0; j < N; ++j) state[j] = 0.0f;

    for (int t = 0; t < T; ++t) {
        const long long o = (((long long)b * T + t) * H + h) * N;
        __syncthreads();
        sr[i]  = r[o + i];
        sw[i]  = __expf(R7_WSCALE / (1.0f + __expf(-wraw[o + i])));
        sk[i]  = k[o + i];
        sav[i] = a[o + i];
        sb[i]  = b_[o + i];
        __syncthreads();

        // THE DELTA-RULE READ, over S_{t-1}, BEFORE the state is advanced.
        float s_a = 0.0f;
#pragma unroll
        for (int j = 0; j < N; ++j) s_a = fmaf(state[j], sav[j], s_a);
        if (sa != nullptr) sa[o + i] = s_a;

        const float vi = v[o + i];
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < N; ++j) {
            float s = fmaf(state[j], sw[j], fmaf(s_a, sb[j], sk[j] * vi));
            acc = fmaf(s, sr[j], acc);                 // y reads the NEW state, unlike v6
            state[j] = s;
        }
        y[o + i] = acc;

        // THE CHUNK BOUNDARY, AND THE LAST STEP WHATEVER t IT FALLS ON. `t == T - 1` is not a
        // special case bolted on: it is what makes a short final chunk legal, and when T is a
        // multiple of C the two conditions coincide on the same step and write once.
        if (ss != nullptr && ((t + 1) % C == 0 || t == T - 1)) {
            float* __restrict__ dst =
                ss + ((((long long)b * H + h) * NC + (t / C)) * N + i) * N;
#pragma unroll
            for (int j = 0; j < N; ++j) dst[j] = state[j];
        }
    }
}

// BACKWARD. Q carries dL/dS_{t-1} back; P_t[i][j] = dL/dS_t is completed from y_t and Q:
//
//     P_t[i][j] = gy_t[i] r_t[j] + Q[j]
//     gsa_t[i]  = sum_j P_t[i][j] b_t[j]                  the delta-rule read's gradient
//     gr_t[j]  += gy_t[i] S_t[i][j]        gb_t[j] += P_t[i][j] sa_t[i]
//     gk_t[j]  += P_t[i][j] v_t[i]         gv_t[i]  = sum_j P_t[i][j] k_t[j]
//     gw_t[j]  += P_t[i][j] S_{t-1}[i][j] w_t[j] W_SCALE sig(1-sig)
//     ga_t[j]  += gsa_t[i] S_{t-1}[i][j]
//     Q[j]      = P_t[i][j] w_t[j] + gsa_t[i] a_t[j]      BOTH routes into S_{t-1}
//
// UNCHANGED FROM THE VERIFIED VERSION except that N is now compile time and the per-step
// vectors are staged. The saved state is still read rather than recomputed; the authors
// recompute from chunk boundaries, which is a memory trade the profile does not need since it
// times the forward and the forward writes nothing under no_grad.
template <int N>
__device__ __forceinline__
void v7_backward(const float* __restrict__ r, const float* __restrict__ wraw,
                 const float* __restrict__ k, const float* __restrict__ v,
                 const float* __restrict__ a, const float* __restrict__ b_,
                 const float* __restrict__ ss, const float* __restrict__ sa,
                 const float* __restrict__ gy, float* __restrict__ gr, float* __restrict__ gw,
                 float* __restrict__ gk, float* __restrict__ gv,
                 float* __restrict__ ga, float* __restrict__ gb,
                 const int T, const int H, const int C) {
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    const int NC = (T + C - 1) / C;

    __shared__ float sr[N], sk[N], sav[N], sb[N], sgm[N], sw[N], siw[N];
    // TWO ROWS PER THREAD, WHICH IS WHAT IT WAS. P is gone -- it is written into Q in place,
    // since P_t is a function of the incoming Q and the outgoing Q is a function of P -- and St
    // takes the register budget P had. Adding a third array at N = 72 or 128 would spill the
    // state row to local memory, which is the exact failure this file was rewritten to fix.
    float Q[N], St[N];
#pragma unroll
    for (int j = 0; j < N; ++j) Q[j] = 0.0f;

    // CHUNKS BACKWARDS, AND WITHIN EACH CHUNK, STEPS BACKWARDS. The stored state is read ONCE
    // per chunk -- the state after the chunk's last step -- and every earlier state in the
    // chunk is recovered from it by inverting the recurrence.
    for (int c = NC - 1; c >= 0; --c) {
        const int hi = ((c + 1) * C < T ? (c + 1) * C : T) - 1;   // last t in this chunk
        const int lo = c * C;
        const float* __restrict__ src =
            ss + ((((long long)b * H + h) * NC + c) * N + i) * N;
#pragma unroll
        for (int j = 0; j < N; ++j) St[j] = src[j];               // S at step `hi`

        for (int t = hi; t >= lo; --t) {
            const long long o = (((long long)b * T + t) * H + h) * N;
            __syncthreads();
            sr[i]  = r[o + i];
            sk[i]  = k[o + i];
            sav[i] = a[o + i];
            sb[i]  = b_[o + i];
            sgm[i] = 1.0f / (1.0f + __expf(-wraw[o + i]));   // sigma, kept for its derivative
            sw[i]  = __expf(R7_WSCALE * sgm[i]);
            // CORRECTLY ROUNDED, NOT APPROXIMATE. --use_fast_math would turn `1.0f / sw[i]`
            // into a fast reciprocal, and this one is applied once per backstep to a quantity
            // the next backstep reads, so its error compounds within the chunk.
            siw[i] = __frcp_rn(sw[i]);
            __syncthreads();

            const float g = gy[o + i], vi = v[o + i], s_a = sa[o + i];
            float gsa = 0.0f, gv_i = 0.0f;
#pragma unroll
            for (int j = 0; j < N; ++j) {
                Q[j] = fmaf(g, sr[j], Q[j]);                 // Q now holds P_t, dL/dS_t
                gsa = fmaf(Q[j], sb[j], gsa);
            }
#pragma unroll
            for (int j = 0; j < N; ++j) {
                const float P = Q[j], sg = sgm[j], w = sw[j], Stj = St[j];
                // THE BACKSTEP, and the only place S_{t-1} comes from now. At t = 0 there is no
                // earlier state: the recurrence starts from zero, and inverting into it would
                // divide a rounding residue by w rather than produce it.
                const float Sp = (t > 0) ? (Stj - fmaf(s_a, sb[j], sk[j] * vi)) * siw[j] : 0.0f;
                atomicAdd(&gr[o + j], g * Stj);
                atomicAdd(&gb[o + j], P * s_a);
                atomicAdd(&gk[o + j], P * vi);
                gv_i = fmaf(P, sk[j], gv_i);
                atomicAdd(&gw[o + j], P * Sp * w * R7_WSCALE * sg * (1.0f - sg));
                atomicAdd(&ga[o + j], gsa * Sp);
                Q[j] = fmaf(P, w, gsa * sav[j]);             // BOTH routes into S_{t-1}
                St[j] = Sp;                                  // and S_{t-1} is next step's S_t
            }
            gv[o + i] = gv_i;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// THE ENTRY POINTS. One pair per head size, because N must be known at compile time. B and N
// are passed but unused -- B is implicit in the grid, N is the template argument -- and kept so
// every family under kernel/cuda takes the same (B, T, H, N) tail. C, the chunk length, follows
// it as a RUNTIME argument and not a second template parameter: nothing unrolls over it, so
// templating would have doubled twelve entry points into twenty-four to no purpose and would
// have frozen a value that is worth being able to sweep.
#define R7_ENTRY(NN)                                                                          \
extern "C" __global__ __launch_bounds__(NN, 2)                                                \
void rwkv7_forward_n##NN(const float* __restrict__ r, const float* __restrict__ wraw,         \
                         const float* __restrict__ k, const float* __restrict__ v,            \
                         const float* __restrict__ a, const float* __restrict__ b_,           \
                         float* __restrict__ y, float* __restrict__ ss,                       \
                         float* __restrict__ sa,                                              \
                         const int B, const int T, const int H, const int N, const int C) {   \
    (void)B; (void)N;                                                                         \
    v7_forward<NN>(r, wraw, k, v, a, b_, y, ss, sa, T, H, C);                                 \
}                                                                                             \
extern "C" __global__ __launch_bounds__(NN, 2)                                                \
void rwkv7_backward_n##NN(const float* __restrict__ r, const float* __restrict__ wraw,        \
                          const float* __restrict__ k, const float* __restrict__ v,           \
                          const float* __restrict__ a, const float* __restrict__ b_,          \
                          const float* __restrict__ ss, const float* __restrict__ sa,         \
                          const float* __restrict__ gy, float* __restrict__ gr,               \
                          float* __restrict__ gw, float* __restrict__ gk,                     \
                          float* __restrict__ gv, float* __restrict__ ga,                     \
                          float* __restrict__ gb,                                             \
                          const int B, const int T, const int H, const int N, const int C) {  \
    (void)B; (void)N;                                                                         \
    v7_backward<NN>(r, wraw, k, v, a, b_, ss, sa, gy, gr, gw, gk, gv, ga, gb, T, H, C);       \
}

R7_ENTRY(16)
R7_ENTRY(32)
R7_ENTRY(48)
R7_ENTRY(60)  // 480/8 -- RWKV-7 at a step-16 parameter match
R7_ENTRY(64)  // 512/8
R7_ENTRY(68)  // 544/8 -- a step-16 match for a 544-wide baseline
R7_ENTRY(70)  // 560/8 -- RWKV-6 at a step-16 parameter match
R7_ENTRY(72)  // 576/8
R7_ENTRY(73)  // 584/8 -- exact fixed-depth match to the 12-layer MIRA-Newton benchmark
R7_ENTRY(80)
R7_ENTRY(96)
R7_ENTRY(128)
