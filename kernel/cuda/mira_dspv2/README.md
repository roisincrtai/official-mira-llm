# kernel/cuda/mira_dspv2 — mira_dspv1's fused scan, rebuilt for throughput

mira_dspv2 is one self-contained package with mira_dspv1's interface: `matrix_lattice_iir_control(x,
control, coefficient_weight, coefficient_bias, spans, gamma, memory_write, valid=None, state=None,
return_state=False, arch=...)`, `matrix_lattice_iir` and every other name mira_dspv1 has. Its
mira_iir.cu holds mira_dspv1's kernels (the chunked forward, the backward, the tree scan, the 2-token
fused scan) and the kernel below, compiled into one cubin; nothing of mira_dspv1 is built or loaded.
It is chosen like any MIRA CUDA package, by `mira_cuda_bin` in a yaml.

`mira_dspv2_fused_scan_k2_m16_r{d_r}_s{mode}_bf16` computes the same function as mira_dspv1's fused scan
(alpha, beta and rho formed from the control on the tensor cores, the lattice recurrence, the
readout and the gate), with four changes that cut the work per token:

* **Tiles of 4 tokens, not 2.** A block is 16 batch rows × 32 channels (16 warps) in two halves
  of 8 rows. Each half forms, stages and synchronises on its own named barrier once per 4 tokens,
  so the per-tile cost (barriers, address arithmetic, staging) is spread over twice the tokens.
* **The halves run half a tile apart.** Half 1 starts with a 2-token tile, so while one half
  forms coefficients on the tensor cores the other runs its recurrence on the FP32 units.
* **memory_read is formed inside the kernel** from the same control, with the same
  `mma.m16n8k16` (bf16 in, fp32 accumulate, one rounding to bf16 as cuBLAS does), so the
  replay launches no separate projection.
* **Fewer instructions per token.** The weight and control fragments come in with `ldmatrix`
  (conflict-free row strides), the bias rides in the accumulator, the readout keeps one sum per
  level and applies rho once, x is folded into the top stage's coefficients, and the token loop
  is compiled twice (with and without a mask) so an unmasked call carries no mask test.

The weight tiles are shared by both halves; the block uses 96.5 KB of shared memory at
d_r = 48 and runs one block per SM. d_r = 16, 32 and 48 are built; d_r = 64 (which does not fit),
training, a split sequence, other dtypes, compute capability below 8.0, and batches too small to
fill the GPU with 16-row blocks run the package's other kernels (the 2-token fused scan, or the
chunked forward and backward). The kernel is built for sm_75 to
sm_121 like mira_dspv1.

```sh
make -f kernel/cuda/mira_dspv2/Makefile universal
make -f kernel/cuda/mira_dspv2/Makefile check        # every kernel against the PyTorch reference on the GPU
python kernel/cuda/mira_dspv2/check.py --cpu         # the dispatch rules and the chunked computation, no GPU
python kernel/cuda/mira_dspv2/check.py --emulate     # the compiled sm_89 SASS run in simulator/nvidia/emulate.py
```

## Expected speed (simulator/nvidia/gpu.py, RTX 6000 Ada, the profiler's max-batch shapes)

| T | B | scan kernel ms, mira_dspv1 | scan kernel ms, mira_dspv2 | mixer M tok/s, mira_dspv1 | mixer M tok/s, mira_dspv2 |
|---|---|---|---|---|---|
| 2,048 | 574 | 52.9 | 39.2 | 9.75 | 11.03 |
| 4,096 | 291 | 57.0 | 42.2 | 9.40 | 10.66 |
| 8,192 | 144 | 65.1 | 48.3 | 9.30 | 10.75 |
| 16,384 | 72 | 65.1 | 48.3 | 8.93 | 10.26 |
| 32,768 | 36 | 65.6 | (2-token fused scan runs) | 8.70 | 8.70 |

1,696 cycles per token per warp against mira_dspv1's 2,287, at the same 16 warps per SM. At
T = 32,768 the 36 rows make 144 blocks for 142 SMs, a second wave of 2 blocks, so that call runs the
2-token fused scan.
