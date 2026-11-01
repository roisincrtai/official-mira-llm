// mamba_s6.cu -- OUR selective scan, so the Mamba baseline needs no external package.
//
//     dA_t = exp(delta_t A)                      (d_in, N) per step
//     dB_t = delta_t B_t                         outer of a channel scalar and an N vector
//     h_t  = dA_t . h_{t-1} + dB_t . u_t         h in R^{d_in x N}
//     y_t  = sum_n h_t[:,n] C_t[n] + D u_t
//     out  = y . SiLU(z)
//
// WHY THIS EXISTS. profiler/baselines.py timed Mamba through the authors' mamba-ssm wheel, and on
// Alliance Canada that wheel cannot be made to work: mamba_ssm 2.2.4+computecanada is a compiled
// extension linked against torch~=2.5.0, the cluster's torch is 2.13.0, and the .so fails with
//
//     undefined symbol: c10::cuda::c10_cuda_check_implementation(int, char const*, ...)
//
// which no install order fixes -- pip said so itself. A benchmark that cannot run is not a
// benchmark. This is ours, it builds from source with nvcc, and it depends on nothing.
//
// THE STATE IS THE WHOLE DIFFERENCE FROM kernel/cuda/mira. MIRA carries ONE SCALAR per channel, so
// mira_scan holds it in a register. Mamba carries N = 16 STATE DIMENSIONS PER CHANNEL -- 16,384
// values per token at d_in = 1024 against MIRA's 512 -- and that is the architecture, not the
// implementation. Here one thread owns one (batch, channel) and keeps all N in REGISTERS:
// N <= 32 fits comfortably, and the (d_in x N) intermediate is never written to memory, which
// is the entire point of the fused kernel this replaces.
//
// LAYOUT, matching mamba_ssm.ops.selective_scan_interface.selective_scan_fn so profiler/baselines
// needs no reshaping:
//     u, delta, z, out   (B, d_in, T)      A  (d_in, N)      D  (d_in)
//     B_, C_             (B, N, T)
// All contiguous float32.
//
// VERIFIED IN NUMPY BEFORE THIS FILE EXISTED: the forward against the definition, and all seven
// gradients against central finite differences at three shapes -- worst 1.0e-09. Two of them
// were WRONG on the first derivation (d/du and d/dB, each off by a factor of u, because the
// write term delta*B*u has each factor exactly once and I had counted u twice). That is what
// finite differences are for.


template <int N>
__device__ __forceinline__
void ms6_fwd(const float* __restrict__ u,        // (B, d, T)
                 const float* __restrict__ dlt,      // (B, d, T)
                 const float* __restrict__ A,        // (d, N)
                 const float* __restrict__ Bm,       // (B, N, T)
                 const float* __restrict__ Cm,       // (B, N, T)
                 const float* __restrict__ Dv,       // (d)
                 const float* __restrict__ z,        // (B, d, T)
                 float* __restrict__ out,            // (B, d, T)
                 float* __restrict__ hs,             // (B, T, d, N) or null -- kept for bwd
                 const int B, const int d, const int T) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * d) return;
    const int c = idx % d;
    const int b = idx / d;

    float h[N], av[N];
    _Pragma("unroll")
        for (int n = 0; n < N; ++n) { h[n] = 0.0f; av[n] = A[(long long)c * N + n]; }
    const float Dc = (Dv != nullptr) ? Dv[c] : 0.0f;

    const long long ub = (long long)b * d * T + (long long)c * T;   // (b, c, .)
    const long long bb = (long long)b * N * T;                      // (b, ., .)

    for (int t = 0; t < T; ++t) {
        const float dt = dlt[ub + t];
        const float ut = u[ub + t];
        float acc = 0.0f;
        _Pragma("unroll")
        for (int n = 0; n < N; ++n) {
            const float dA = __expf(dt * av[n]);
            // dB_t . u_t = delta_t * B_t[n] * u_t -- each factor once.
            h[n] = fmaf(dA, h[n], dt * Bm[bb + (long long)n * T + t] * ut);
            acc = fmaf(h[n], Cm[bb + (long long)n * T + t], acc);
            if (hs != nullptr)
                hs[((((long long)b * T + t) * d) + c) * N + n] = h[n];
        }
        const float y = fmaf(Dc, ut, acc);
        const float zz = z[ub + t];
        out[ub + t] = y * (zz / (1.0f + __expf(-zz)));              // gated by SiLU(z)
    }
}

// BACKWARD. G_t = dL/dh_t accumulated from the output and from the future:
//
//     gy_t  = go_t SiLU(z_t) ;  gz_t = go_t y_t SiLU'(z_t)
//     gC_t  = sum_d gy_t h_t          gD = sum_{b,t} gy_t u_t
//     G_t  += gy_t C_t                (then G_{t-1} <- G_t . dA_t)
//     gA   += G_t h_{t-1} dA_t delta_t        gdelta_t += (G_t h_{t-1} dA_t A) . 1
//     gu_t += (G_t delta_t B_t) . 1           gB_t += sum_d G_t delta_t u_t
//     gdelta_t += (G_t B_t u_t) . 1
//
// gA, gB, gC and gD are reduced across (b, t) or (d) by atomicAdd, so they are NOT bit
// reproducible; the per-thread partials are accumulated in registers first so there is one
// atomic per (thread, n) rather than one per element.
template <int N>
__device__ __forceinline__
void ms6_bwd(const float* __restrict__ u,
                  const float* __restrict__ dlt,
                  const float* __restrict__ A,
                  const float* __restrict__ Bm,
                  const float* __restrict__ Cm,
                  const float* __restrict__ Dv,
                  const float* __restrict__ z,
                  const float* __restrict__ hs,      // (B, T, d, N) from the forward
                  const float* __restrict__ go,      // (B, d, T)
                  float* __restrict__ gu,
                  float* __restrict__ gdlt,
                  float* __restrict__ gA,            // (d, N)      accumulated
                  float* __restrict__ gB,            // (B, N, T)   accumulated
                  float* __restrict__ gC,            // (B, N, T)   accumulated
                  float* __restrict__ gD,            // (d)         accumulated, or null
                  float* __restrict__ gz,
                  const int B, const int d, const int T) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * d) return;
    const int c = idx % d;
    const int b = idx / d;

    float G[N], av[N], acc_A[N];
    _Pragma("unroll")
        for (int n = 0; n < N; ++n) { G[n] = 0.0f; acc_A[n] = 0.0f; av[n] = A[(long long)c * N + n]; }
    const float Dc = (Dv != nullptr) ? Dv[c] : 0.0f;
    float acc_D = 0.0f;

    const long long ub = (long long)b * d * T + (long long)c * T;
    const long long bb = (long long)b * N * T;

    for (int t = T - 1; t >= 0; --t) {
        const float ut = u[ub + t];
        const float dt = dlt[ub + t];
        const float zz = z[ub + t];
        const float sd = 1.0f / (1.0f + __expf(-zz));
        const float sz = zz * sd;                                   // SiLU(z)

        // y_t has to be recomputed to differentiate the gate; h_t is read back from hs.
        float yv = Dc * ut;
        _Pragma("unroll")
        for (int n = 0; n < N; ++n)
            yv = fmaf(hs[((((long long)b * T + t) * d) + c) * N + n],
                      Cm[bb + (long long)n * T + t], yv);
        const float g = go[ub + t];
        gz[ub + t] = g * yv * (sd * (1.0f + zz * (1.0f - sd)));
        const float gy = g * sz;

        acc_D += gy * ut;
        float gu_t = Dc * gy, gd_t = 0.0f;
        _Pragma("unroll")
        for (int n = 0; n < N; ++n) {
            const long long hi = ((((long long)b * T + t) * d) + c) * N + n;
            const float ht = hs[hi];
            const float hprev = (t > 0) ? hs[hi - (long long)d * N] : 0.0f;
            const float Cn = Cm[bb + (long long)n * T + t];
            const float Bn = Bm[bb + (long long)n * T + t];
            atomicAdd(&gC[bb + (long long)n * T + t], gy * ht);
            G[n] += gy * Cn;                                        // dL/dh_t, complete
            const float dA = __expf(dt * av[n]);
            const float gdA = G[n] * hprev;
            acc_A[n] = fmaf(gdA * dA, dt, acc_A[n]);
            gd_t = fmaf(gdA * dA, av[n], gd_t);                     // through dA
            gd_t = fmaf(G[n] * Bn, ut, gd_t);                       // through dB
            gu_t = fmaf(G[n] * dt, Bn, gu_t);
            atomicAdd(&gB[bb + (long long)n * T + t], G[n] * dt * ut);
            G[n] *= dA;                                             // hand down to t-1
        }
        gu[ub + t] = gu_t;
        gdlt[ub + t] = gd_t;
    }
    _Pragma("unroll")
        for (int n = 0; n < N; ++n) atomicAdd(&gA[(long long)c * N + n], acc_A[n]);
    if (gD != nullptr) atomicAdd(&gD[c], acc_D);
}

// ---------------------------------------------------------------------------------------------
// THE ENTRY POINTS. One pair per state width, because N must be known at compile time -- it was
// a runtime loop bound, so nvcc could not keep h[] and av[] in registers and spilled the whole
// per-channel state to LOCAL MEMORY, which is HBM. One thread per (batch, channel) is only
// worth doing if that thread's N state values are in registers. The wrapper picks the name;
// N = 16 is Mamba's default d_state and the only one this profile uses.
//
// `extern "C"` so the driver can find them by name in the cubin.
#define MS6_ENTRY(NN)                                                                         \
extern "C" __global__                                                                         \
void ms6_forward_n##NN(const float* __restrict__ u, const float* __restrict__ dlt,            \
                       const float* __restrict__ A, const float* __restrict__ Bm,             \
                       const float* __restrict__ Cm, const float* __restrict__ Dv,            \
                       const float* __restrict__ z, float* __restrict__ out,                  \
                       float* __restrict__ hs,                                                \
                       const int B, const int d, const int T, const int N) {                  \
    (void)N;                                                                                  \
    ms6_fwd<NN>(u, dlt, A, Bm, Cm, Dv, z, out, hs, B, d, T);                                  \
}                                                                                             \
extern "C" __global__                                                                         \
void ms6_backward_n##NN(const float* __restrict__ u, const float* __restrict__ dlt,           \
                        const float* __restrict__ A, const float* __restrict__ Bm,            \
                        const float* __restrict__ Cm, const float* __restrict__ Dv,           \
                        const float* __restrict__ z, const float* __restrict__ hs,            \
                        const float* __restrict__ go, float* __restrict__ gu,                 \
                        float* __restrict__ gdlt, float* __restrict__ gA,                     \
                        float* __restrict__ gB, float* __restrict__ gC,                       \
                        float* __restrict__ gD, float* __restrict__ gz,                       \
                        const int B, const int d, const int T, const int N) {                 \
    (void)N;                                                                                  \
    ms6_bwd<NN>(u, dlt, A, Bm, Cm, Dv, z, hs, go, gu, gdlt, gA, gB, gC, gD, gz, B, d, T);     \
}

MS6_ENTRY(8)
MS6_ENTRY(16)     // Mamba's default d_state, and the only one this profile uses
MS6_ENTRY(32)
MS6_ENTRY(64)
