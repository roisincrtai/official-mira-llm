"""MIRA CUDA v7 (mira-dsp): mira_v5_stable with the fused scan's x / gamma prefetch fixed.

WHAT CHANGED FROM kernel/cuda/mira_v5_stable (2026-10-12), AND NOTHING ELSE (the math is v5's):
  * fused_scan carries the next tokens' x and gamma as raw bf16 bits, FUSED_AHEAD = 2 tokens
    ahead, and widens them where they are used. v5 widened them at the load, and the conversion's
    scoreboard wait cost every warp a global-memory round trip per token. simulator/nvidia (a
    cycle-level model driven by the compiled SASS, calibrated on v5 and v6) predicts 1.21-1.25x
    v5's scan: ~9.9 M tok/s at T = 2048 on the profiler's max-batch shape.

mira_v5_stable's own description follows.

MIRA CUDA v5 (mira-dsp): mira_v4 with a small, branch-free coefficient-forming inference scan.

WHAT CHANGED FROM kernel/cuda/mira_v4 (2026-10-12), AND NOTHING ELSE (the math is v4's):
  * fused_scan is one kernel per (d_r, stage map), mira_fused_scan_k2_m16_r{d_r}_s{mode}_bf16 with
    mode the ARCH_STAGE value: the stage map is a compile-time constant, so the token loop holds
    only the selected map (v4 carried all 15 maps and the branches between them in every token
    step, ~14 KB of SASS around ~450 executed instructions; v5's loop is ~6 KB).
  * a (row, channel)'s seven coefficients are one 16-byte chunk of a swizzled shared tile
    (chunk row * 32 + (channel ^ (row & 7))): one conflict-free LDS.128 per token, and the forming
    step's stores are conflict-free (v4: 8-way).
  * the forming step interleaves the warp's four channels' mma chains, and stages the next tile
    (double-buffered control tile) while they run.
  * fused_scan_timing: the _timed kernels (d_r = 48) record a per-warp cycle breakdown, which the
    profiler stores.

mira_v4's own description follows.

MIRA CUDA v4 (mira-dsp): mira_v3treescan, plus an inference scan that forms its own coefficients.

WHAT CHANGED FROM kernel/cuda/mira_v3treescan (2026-10-11), AND NOTHING ELSE:
  * matrix_lattice_iir_control: the mixer forward from the d_r-wide control vector. At inference,
    one chunk, bf16 and compute capability 8.0+, fused_scan (mira_iir.cu) forms alpha, beta and rho
    on the tensor cores from the control and the coefficient_projection weights instead of reading
    them from HBM; only memory_read is projected outside. Every other call projects the
    coefficients exactly as before and runs v3's matrix_lattice_iir. The function is v3's; a
    fused coefficient differs only in the order of its d_r-term sum before the same bf16 rounding.
  * model/mira_mixer.py calls it when the selected MIRA CUDA package provides it.

mira_v3treescan's own description follows.

MIRA CUDA v3 tree scan (mira-dsp): mira_v2 with the chunk composition done by one tree scan.

WHAT CHANGED FROM kernel/cuda/mira_v2 (2026-10-11), AND NOTHING ELSE:
  * p2, p2i, b2 and the 16-chunk affine summary/replay (chunks >= 128) are replaced by one kernel,
    mira_tree_scan_k{K}_m{d_m}, launched once for any chunk count, forward and backward
    (_compose_boundaries, _tree_plan): each (sequence, channel, slice of memory columns) gets
    `workers` threads that fold their run of chunks, scan the folded maps (Kogge-Stone in the warp,
    then across warps), and replay; workers come from the resident threads of this GPU;
  * p1i never writes the last chunk's summary at any chunk count (the tree never reads it);
  * every chunk-length rule is v2's, so profiling v2 against v3 isolates the composition.
What the kernels compute is v2's; the tree reassociates the composition's floating-point sums.

mira_v2's own description follows.

MIRA CUDA v2 (mira-dsp): v1's kernels, with every chunk length derived from the GPU it runs on.

WHAT CHANGED FROM kernel/cuda/mira_v1 (2026-10-11), AND NOTHING ELSE:
  * the forward token chunk, in training and in inference, comes from one closed-form rule over the
    shape and the queried hardware (_forward_chunk), in place of v1's 512-block / 128-token
    training rule and its inference rule built on 2 and 5 assumed blocks per SM;
  * the backward fine chunk comes from the shared memory this GPU grants b3 (_fine_chunk), in
    place of BACKWARD_MAP's fixed chunk column sized against 48 KiB;
  * the resident blocks of every launched kernel are read from the driver's occupancy query once
    per (device, kernel, block, shared size) and cached; the __launch_bounds__ minimum is used when
    the driver will not say;
  * p2 issues the next chunk's loads before composing the current one (mira_iir.cu);
  * the cubin is built for every CUDA GPU from Turing on (Makefile).
What the kernels compute is v1's: only where the token axis is cut changes, and with it the
association order of floating-point sums.
"""
from __future__ import annotations

import math
import os

from .. import driver as D


CUBIN = "mira_dspv1/mira_iir.cubin"
ORDER = 4
# THE ORDERS THE CUBIN CARRIES, and they must cover ablation/run.py's
# LATTICE_ORDER_SWEEP = (1, 2, 4, 8): MIRA declines rather than falling back to the
# PyTorch reference, so an order missing here is a hard failure at the first step of a
# 10,000-step ablation variant, not a slow path.
ORDERS = (1, 2, 4, 8)
MEMORY_DIMENSIONS = (8, 16, 32, 64)
CHANNEL_TILE = 128
CHANNEL_TILES = {(2, 64): 256}
# THE BLOCKS PER SM EACH KERNEL IS COMPILED FOR (the __launch_bounds__ minimum in mira_iir.cu): the
# occupancy used when the driver will not report one -- a guaranteed lower bound, never a guess
# upward. p1 and p3 are compiled for 3 blocks per SM when a block is a third of the SM's threads or
# less and K <= 2 (MIRA_MIN_BLOCKS); 2 is the bound every one of them holds.
LAUNCH_BOUNDS_BLOCKS = {"p1": 2, "p3": 2, "p1i": 2, "p3i": 5, "b3": 1, "tree": 1}
# THE TREE SCAN BLOCK (mira_iir.cu TREE_THREADS) and its memory columns per thread, V = min(d_m, 8/K).
TREE_THREADS = 256
TREE_MAX_WORKERS = 256
# THE SPLIT MUST PAY FOR ITSELF BY MORE THAN AN EIGHTH. A split adds two launches (p1 and the
# composition) that the time model does not count, and the model is least reliable near a tie.
SPLIT_MARGIN = 1.125
WARP_WIDTH = 32
TARGET_THREADS = 16 * WARP_WIDTH
COMPOSE_THREADS = 2 * WARP_WIDTH
BACKWARD_THREADS = 8 * WARP_WIDTH
BACKWARD_SEGMENT = {8: 8, 16: 8, 32: 8, 64: 8}
REDUCTION_THREADS = 256
ALPHA_TILE = 16


def _scan_variants(memory, order):
    # THE LANE COUNT IS max(K, d_m / 4) AND K IS THE ARGUMENT, NOT THE MODULE CONSTANT. Reading
    # the module-level ORDER here made every order at d_m = 16 take four lanes, and the K = 8
    # kernel then trained on uninitialised memory in silence: p1 declares float basis[N], seeds
    # only `if (lane < N)` and stores phi[(plane*N + row)*N + lane], so with N = 8 on L = 4 lanes
    # phi columns 4..7 of a torch.empty buffer were never written, and p2's
    # __shfl_sync(mask, owned, column, 4) wrapped columns 4..7 back onto lanes 0..3. The floor at
    # K is therefore load-bearing, and mira_iir.cu now carries a static_assert(LANES >= N) so a
    # future mismatch is a compile error instead of garbage gradients. Only K = 8 at d_m = 16
    # moves under this fix, from 4 lanes x 128 channels to 8 x 64; K = 1, 2 and 4 keep the
    # 4 x 128, 8 x 64 and 16 x 32 they already had at d_m 16, 32 and 64, because for those orders
    # max(K, d_m / 4) equals d_m / 4 at every memory dimension.
    if (order, memory) == (2, 64):
        return ((16, 16),)
    lanes = max(order, memory // 4)
    return ((lanes, TARGET_THREADS // lanes),)


def _inference_scan_variants(memory, order):
    if memory == 16 and order == 2:
        return ((2, 192),)
    if memory == 64 and order == 2:
        return ((8, 32),)
    return _scan_variants(memory, order)


def _inference_replay_variants(memory, order):
    if memory == 16 and order == 2:
        return ((1, 128),)
    if memory == 64 and order == 2:
        return ((4, 32),)
    return _inference_scan_variants(memory, order)


def _phase_one_variants(memory, order):
    if (order, memory) == (2, 64):
        return ((16, 24),)
    return _scan_variants(memory, order)


def _compose_variants(memory, order):
    """The lanes and channel group of the single-token step (mira_v3treescan: p2 and b2, for which
    this was written, are replaced by the tree scan; the reasoning below is v2's).

    THEIR GRID IS batch x ceil(d_e / G) AND NOTHING ELSE, because each block walks every chunk
    of its channels serially. At the scan's G = 128 that is 12 groups at d_e = 1,536, so batch 2
    launches 24 blocks and batch 6 launches 72 -- against 72 SMs, a third of the card at
    profiling shapes and one block per SM at training shapes, on a kernel that is a serial chain
    of dependent global loads and has nothing else resident to hide them behind.

    THE GROUP IS FREE TO NARROW BECAUSE THE LAYOUT DOES NOT CONTAIN IT. scan_phase_two indexes
    phi, carry and checkpoint by plane = (batch, chunk, channel), and G appears only in
    `groups = ceil(channels/G)` and `channel = group*G + local_channel` -- so a narrower group
    is the same arithmetic on the same bytes, in more blocks. The lane count is NOT free: it
    divides the memory axis and carries the phi column shuffle, so it stays what the scan uses.

    TWO WARPS PER BLOCK, DERIVED AND NOT SEARCHED. COMPOSE_THREADS / L holds the block at 64
    threads at every built shape, which multiplies the block count by eight and leaves the total
    warp count unchanged -- the same work, packaged so it reaches every SM."""
    lanes = max(order, memory // 4)
    return ((lanes, COMPOSE_THREADS // lanes),)


def _backward_variants(memory):
    lanes = min(WARP_WIDTH, memory)
    primary = min(BACKWARD_THREADS // lanes, BACKWARD_THREADS // memory)
    return ((lanes, primary), (lanes, primary // 2))


# THE MEMORY WIDTHS EACH ORDER IS BUILT FOR, AND NOTHING WIDER. Orders 2 and 4 keep all three
# widths, exactly as before. Orders 1 and 8 exist for ablation/run.py's LATTICE_ORDER_SWEEP, and
# that sweep runs the mira-s scheme, whose memory_dimension is 16; d_m 32 and 64 at those two
# orders would be compiled and never launched. Building them took the file from 105 instantiations
# to 243 and the compile from seconds to minutes. A shape with no row here raises at the launcher
# rather than falling back, which is what a missing instantiation must do.
ORDER_MEMORIES = {
    1: (8, 16),
    2: MEMORY_DIMENSIONS,
    4: MEMORY_DIMENSIONS,
    8: (8, 16),
}


def _built_pairs():
    """Every (order, d_m) the cubin holds, in report and enumeration order.

    THE CROSS PRODUCT OF ORDERS AND MEMORY_DIMENSIONS IS NOT THAT SET and must not be used as
    though it were: orders 1 and 8 are built at d_m 16 alone, so crossing the two tuples names
    four shapes that have no kernel and no BACKWARD_MAP row. report() did exactly that and raised
    KeyError (32, 1) inside `make check`, after the cubin had already built successfully.
    """
    return tuple((order, memory) for order in ORDERS for memory in ORDER_MEMORIES[order])


def _b3_variants(order, memory):
    """The b3 shapes built for this (order, d_m), which is now the same two for every order."""
    if memory == 8:
        return ((8, {1: 32, 2: 16, 4: 8, 8: 4}[order]),)
    if (order, memory) == (2, 64):
        return ((32, 16),) + _backward_variants(memory)[:2]
    del order
    return _backward_variants(memory)[:2]


FNS = tuple(
    f"mira_matrix_scan_{phase}_k{order}_m{memory}_l{lanes}_g{group}_{suffix}"
    for order in ORDERS
    for memory in ORDER_MEMORIES[order]
    for lanes, group in _scan_variants(memory, order)
    for phase in ("p1", "p3", "b1")
    for suffix in ("f32", "bf16", "f16")
) + tuple(
    f"mira_matrix_scan_b3_k{order}_m{memory}_l{lanes}_g{group}_{suffix}"
    for order in ORDERS
    for memory in ORDER_MEMORIES[order]
    for lanes, group in _b3_variants(order, memory)
    for suffix in ("f32", "bf16", "f16")
) + tuple(
    f"mira_matrix_scan_b4_k4_{suffix}" for suffix in ("f32", "bf16", "f16")
)
FNS += tuple(
    f"mira_matrix_scan_p1i_k2_m16_l2_g192_{suffix}"
    for suffix in ("f32", "bf16", "f16")
) + tuple(
    f"mira_matrix_scan_p3i_k2_m16_l1_g128_{suffix}"
    for suffix in ("f32", "bf16", "f16")
)
FNS += tuple(
    f"mira_matrix_scan_p1i_k2_m64_l8_g32_{suffix}"
    for suffix in ("f32", "bf16", "f16")
) + tuple(
    f"mira_matrix_scan_p3i_k2_m64_l4_g32_{suffix}"
    for suffix in ("f32", "bf16", "f16")
)
FNS += tuple(
    f"mira_matrix_scan_p1_k{order}_m{memory}_l{lanes}_g{group}_{suffix}"
    for order in ORDERS
    for memory in ORDER_MEMORIES[order]
    for lanes, group in _phase_one_variants(memory, order)
    if (lanes, group) not in _scan_variants(memory, order)
    for suffix in ("f32", "bf16", "f16")
)
FNS += tuple(
    f"mira_matrix_step_k{order}_m{memory}_{suffix}"
    for order, memory in _built_pairs() for suffix in ("f32", "bf16", "f16")
)
FNS += tuple(f"mira_tree_scan_k{order}_m{memory}" for order, memory in _built_pairs())
SUFFIX = {"torch.float32": "f32", "torch.bfloat16": "bf16", "torch.float16": "f16"}
# NO ACTIVATION AND NO tau CROSS THIS BOUNDARY. alpha reaches the stage through the Cayley
# rotation and beta through r = 1 / (1 + beta^2) (arch=cauchy_x_mobius), or through the
# hyperbolic rotation sin = tanh(alpha), cos = sech(alpha) (arch=cauchy_x_hyperbolic), or through the
# identity rotation theta = alpha, sin(alpha), cos(alpha) (arch=cauchy_x_identity); all are bounded for every real argument, so
# neither needs a clamp and the kernel carries no activation selector. silu_cap, tanh and nsilu
# were clamps on a reflection coefficient this stage does not have, and tau was tanh's alone.
# They were accepted and discarded at first use -- coefficient() opened with (void)tau;
# (void)activation -- so removing them takes one pointer and one int out of every launch, out of
# eight kernel signatures and out of the autograd save list.
# THE CANONICAL STAGE IS MODE 3 IN mira_iir.cu: descending k, the level formed from the
# incoming U, and state[0] closed with U_t^(0). It is the only mode the cubin instantiates.
CANONICAL_STAGE = 3
# The operator applied at every stage k = 1..K of the lattice.  Mode 0 is the lattice stage
# that subtracts alpha times the delayed level and re-injects the UPDATED wave; mode 1 is the rotation
# stage that maps (wave, delayed level) through [[cos, -sin], [sin, cos]] of the same alpha and
# re-injects the INCOMING wave.  Everything else -- the chunking, the delay line, the K+1 taps,
# the readout and the persistent state -- is shared, so the mode is one runtime scalar and not a
# second set of entry points.  Mode 1 stages cos and sin together, so its alpha cache is twice as
# wide as mode 0's.
# Mode 2 is the local rotation: the same 2x2 as mode 1, but each stage reads its OWN delayed
# level rather than the one below it, so both indices ascend together and the chain traverses k = 1..K
# from the input.  Its persistent state is V^(1..K) rather than V^(0..K-1), its stage-zero tap is
# V_t^(0) = U_t^(0), the current write, and it stages the same cos/sin pair mode 1 does, so every
# stored shape and the alpha cache width are unchanged.
# Mode 3 is the decoupled shifting stage: the traversal, the closure and the persistent state
# V^(0..K-1) of mode 1, with the 2x2 a rotation scaled by its own radius,
#     (U_t^(k-1), V_t^(k)) = r_t^(k) R(theta_t^(k)) (U_t^(k), V_{t-1}^(k-1)),
#     sin(theta) = (1 - alpha^2)/(1 + alpha^2),  cos(theta) = 2 alpha/(1 + alpha^2),
#     r = 1 / (1 + beta^2),
# so the K=1 recurrent pole is -r sin(theta), with alpha setting the angular factor and beta
# setting r. raw_beta is a second [B, T, C, K] projection of the same bottleneck. It stages cos,
# sin and r, so its alpha tile is three times mode 0's, and it is the only mode that reads
# raw_beta or writes grad_beta; under modes 0, 1 and 2 both pointers are null.
# Mode 6 is mode 3 with the hyperbolic rotation sin = tanh(alpha), cos = sech(alpha), and mode 7
# is mode 3 with the identity rotation theta = alpha, sin(alpha), cos(alpha).
# Modes 8, 9 and 10 use the gaussian radius r = exp(-beta^2) with the mobius, hyperbolic and identity rotations.
# Modes 11, 12 and 13 use the expexp radius r = exp(-exp(beta)) with those three rotations.
# Modes 14, 15 and 16 use the expsoftplus radius r = exp(-softplus(beta)) with those three rotations.
# Modes 17, 18 and 19 use the hill radius r = beta^2 / (1 + beta^2) with those three rotations.
# It runs mode 3's compiled bodies with a runtime value read where the coefficients are staged
# and where grad_beta is stored, so they add no kernel, and their spans are mode 3's.
# The read direction is per stage under every mode: memory_read is [B, T, K+1, M] and tap k
# contracts nu^(k), so its gradient is [B, T, K+1, M] too.  rho carries no activation.
STAGE = {"lattice": 0, "rotation": 1, "local_rotation": 2, "decoupled_shift": 3}
STAGE_SPAN = {0: 1, 1: 2, 2: 2, 3: 4, 6: 4, 7: 4, 8: 4, 9: 4, 10: 4, 11: 4, 12: 4, 13: 4, 14: 4, 15: 4, 16: 4, 17: 4, 18: 4, 19: 4}
SCAN_STAGE_SPAN = {0: 1, 1: 2, 2: 2, 3: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 2, 16: 2, 17: 2, 18: 2, 19: 2}
# THE MODE OF EACH ARCH: cauchy_x_mobius is the canonical stage.
ARCH_STAGE = {"cauchy_x_mobius": CANONICAL_STAGE, "cauchy_x_hyperbolic": 6, "cauchy_x_identity": 7,
              "gaussian_x_mobius": 8, "gaussian_x_hyperbolic": 9, "gaussian_x_identity": 10,
              "expexp_x_mobius": 11, "expexp_x_hyperbolic": 12, "expexp_x_identity": 13,
              "expsoftplus_x_mobius": 14, "expsoftplus_x_hyperbolic": 15, "expsoftplus_x_identity": 16,
              "hill_x_mobius": 17, "hill_x_hyperbolic": 18, "hill_x_identity": 19}
USED = {"forward": 0, "backward": 0}
FUNCTION = None
REPORTED = set()


def _row_stride(tensor, tokens):
    if tensor.stride(2) != 1:
        return None
    row = int(tensor.stride(1))
    if int(tensor.stride(0)) != tokens * row:
        return None
    return row


def dense_tail_row_stride(tensor, tokens):
    expected = 1
    for axis in range(tensor.dim() - 1, 1, -1):
        if int(tensor.stride(axis)) != expected:
            return None
        expected *= int(tensor.shape[axis])
    row = int(tensor.stride(1))
    if int(tensor.stride(0)) != tokens * row:
        return None
    return row


def status(device=None):
    try:
        import torch
        if not torch.cuda.is_available():
            return False, D.why()
        index = torch.cuda.current_device() if device is None else int(device)
        module = D.module_for(CUBIN, index, FNS)
    except Exception:
        return False, D.why()
    return (True, module["__path__"]) if module else (False, D.why())


def usage():
    return dict(USED)


def report(index=None):
    """Print how the problem maps onto the grid, the block, the warp and the lane. Once per device."""
    if os.environ.get("MIRALLM_CUDA_REPORT",
                      os.environ.get("NULLGPT_CUDA_REPORT", "1")) == "0":
        return False
    if not D.main_process():
        return False
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        index = torch.cuda.current_device() if index is None else int(index)
        module = D.module_for(CUBIN, index, FNS)
    except Exception:                                          # noqa: BLE001
        return False
    if module is None or index in REPORTED:
        return False
    REPORTED.add(index)
    ctx = module["__ctx__"]
    sms = D.sm_count(index)
    tag = "mira-dsp"

    def resident(name, block, shared=0):
        try:
            value = D.resident_blocks(module[name], ctx, index, block, shared)
        except Exception:                                      # noqa: BLE001
            return "unavailable"
        if not value or not sms:
            return "unavailable"
        per_sm = value // sms
        return f"{per_sm} blocks/SM, {per_sm * block // WARP_WIDTH} warps/SM"

    pairs = _built_pairs()
    widths = ", ".join(
        "K=%d: %s" % (order, "/".join(str(m) for m in ORDER_MEMORIES[order])) for order in ORDERS)
    lines = [
        f"[{tag}] cuda mapping on cuda:{index} -- {module['__path__']}",
        f"[{tag}]   device                    {sms} SMs, warp {WARP_WIDTH} threads",
        f"[{tag}]   lattice                   orders K={', '.join(str(o) for o in ORDERS)},"
        f" K+1 ladder levels per"
        f" memory slot, memory dimensions {widths}",
        f"[{tag}]   token tile                derived from this GPU (mira_v2): the backward chunk is"
        f" the largest power of two whose b3 shared block fits this GPU, at most one b3 wave long,"
        f" {', '.join(f'K={o} d_m={m} up to {_fine_chunk_device(index, o, m, 2)[0]}' for o, m in pairs)}"
        f" tokens at bf16; shared memory directions are staged {ALPHA_TILE} tokens at a time",
        f"[{tag}]   forward chunks            training and inference: the chunk count that puts the"
        f" grid on whole waves of this GPU's resident blocks, W = max(ceil(x), floor(x sqrt(T/c0))),"
        f" at most sqrt(T) chunks, and one chunk unless the split is faster by more than"
        f" {SPLIT_MARGIN - 1:.3f}; fine FP32 transitions and checkpoints are retained for backward",
        f"[{tag}]   stage operator            the canonical stage, and the only one compiled:"
        f" (U_t^(k-1), V_t^(k)) = r R(theta) (U_t^(k), V_(t-1)^(k-1)) for k = K..1, with the"
        f" shifted Cayley rotation sin(theta) = (1 - alpha^2)/(1 + alpha^2), cos(theta) = 2 alpha/(1 +"
        f" alpha^2) and r = 1 / (1 + beta^2); it stages cos, sin and the radius, so its alpha"
        f" tile is three values wide. cauchy_x_hyperbolic runs the same compiled stage as mode 6,"
        f" with sin(theta) = tanh(alpha), cos(theta) = sech(alpha), and cauchy_x_identity as"
        f" mode 7, with theta = alpha; modes 8, 9 and 10 are the same three rotations with the"
        f" gaussian radius r = exp(-beta^2), modes 11, 12 and 13 with the expexp radius"
        f" r = exp(-exp(beta)), modes 14, 15 and 16 with the expsoftplus radius"
        f" r = exp(-softplus(beta)), and modes 17, 18 and 19 with the hill radius"
        f" r = beta^2 / (1 + beta^2)",
        f"[{tag}]   grid                      one block per (sequence, chunk, channel group);"
        f" phase p1 scans each chunk from a zero carry, the tree scan composes the carries"
        f" along the chunk axis, p3 replays the chunk from its true carry",
        f"[{tag}]   forward scan p1/p3 and backward scan b1 -- lanes divide the memory"
        f" axis, the channel group divides d_e",
    ]
    # THE SCAN MAPPING IS PER (K, d_m), NOT PER d_m, because lanes = max(K, d_m / 4) reads the
    # order: at d_m = 16 the K = 8 row is 8 x 64 while K = 1, 2 and 4 are 4 x 128. Reporting one
    # line per memory dimension would have printed a mapping no K = 8 launch uses.
    for order, memory in pairs:
        for lanes, group in _scan_variants(memory, order):
            block = lanes * group
            lines.append(
                f"[{tag}]     K={order} d_m={memory:<3}            {lanes:>2} lanes x"
                f" {group:>3} channels = {block:>3} threads"
                f" ({block // WARP_WIDTH} warps); one lane carries"
                f" {memory // lanes} memory slots x K+1 levels;"
                f" {resident(_scan_name('p1', order, memory, lanes, group, 'bf16'), block)}")
    lanes, group = _inference_scan_plan(2, 16)
    block = lanes * group
    lines.append(
        f"[{tag}]     K=2 d_m=16 inference    {lanes:>2} lanes x"
        f" {group:>3} channels = {block:>3} threads"
        f" ({block // WARP_WIDTH} warps); one lane carries"
        f" {16 // lanes} memory slots x K+1 levels;"
        f" {resident(_forward_scan_name('p1', 2, 16, lanes, group, 'bf16', False), block)}")
    lanes, group = _inference_replay_plan(2, 16)
    block = lanes * group
    lines.append(
        f"[{tag}]     K=2 d_m=16 inference replay"
        f" {lanes:>2} lanes x"
        f" {group:>3} channels = {block:>3} threads"
        f" ({block // WARP_WIDTH} warps); one lane carries"
        f" {16 // lanes} memory slots x K+1 levels;"
        f" {resident(_forward_scan_name('p3', 2, 16, lanes, group, 'bf16', False), block)}")
    lines.append(
        f"[{tag}]   carry composition -- one tree scan per launch at any chunk count"
        f" (mira_v3treescan): {TREE_THREADS} threads per block; per (sequence, channel, V memory"
        f" columns) `workers` threads fold, scan and replay; backward reads transposed transitions")
    for order, memory in pairs:
        columns = _tree_columns(order, memory)
        lines.append(
            f"[{tag}]     K={order} d_m={memory:<3}            V={columns} memory columns per"
            f" thread, {memory // columns} slices;"
            f" {resident(_tree_name(order, memory), TREE_THREADS)}")
    lines.append(
        f"[{tag}]   backward readout b3 -- one block per (sequence, chunk, channel tile),"
        f" replayed in segments of {BACKWARD_SEGMENT[MEMORY_DIMENSIONS[0]]} tokens; its shared"
        f" budget follows the chunk and is fixed by the mapping")
    lines.append(
        f"[{tag}]   single-token decoding     direct state/readout kernel, 64 threads per block;"
        f" {', '.join(f'K={o} d_m={m}: {max(o, m // 4)} lanes x {64 // max(o, m // 4)} channels' for o, m in pairs)}")
    lines.append(
        f"[{tag}]   backward reduction        d_m=16 sums channel pairs within each warp and"
        f" reduces two tokens per barrier pair; d_m=32/64 retain single-token reduction")
    # b3 IS REPORTED PER (K, d_m) TOO, and only for the channel groups that shape is built for.
    # _b3_variants gives K = 8 the single narrower group BACKWARD_MAP names for it, because the
    # two wider ones do not fit b3's shared budget at that order and are not instantiated.
    for order, memory in pairs:
        for lanes, group in _b3_variants(order, memory):
            block = lanes * group
            lines.append(
                f"[{tag}]     K={order} d_m={memory:<3}            {lanes:>2} lanes x {group:>2}"
                f" channels = {block:>3} threads ({block // WARP_WIDTH} warps)")
    lines.append(
        f"[{tag}]   channel-tile reduction b4  {REDUCTION_THREADS} threads per block, summing the"
        f" write and read gradients over the channel tiles of {CHANNEL_TILE} channels")
    lines.append(
        f"[{tag}]   the variant launched is fixed by closed-form rules, not searched for: the widest"
        f" lanes the memory axis allows, the BACKWARD_MAP channel group (or the narrower compiled"
        f" one when it leaves fewer than 16 tokens), and the token chunks above")
    print("\n".join(lines), flush=True)
    return True


# THE DYNAMIC-SHARED CEILING OF A KERNEL ONLY RISES. cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)
# is a ceiling on the CUfunction, and D.prefer_shared sets it once per (function, size) and caches
# that it did. v2 launches one kernel at several sizes -- the backward chunk follows the GPU and the
# shape, and b3's shared block follows the chunk -- so setting a smaller size after a larger one
# LOWERED the ceiling, and the larger size, already cached, was never set again:
#
#     mira_matrix_scan_b3_k2_m64_l32_g16_f32 rc=1 INVALID_VALUE (block 512, shared 99840)
#
# after a 54,784 B launch of the same kernel (make check, _fine_case at chunk 16 then 32). Every
# size asked of a kernel therefore sets the largest size asked of it so far, never less; every size
# is within the GPU's opt-in, because every one of them comes from _b3_budget or a fixed tile.
_SHARED_CEILING = {}


def _prefer_shared(module, name, shared):
    if not shared:
        return
    key = (id(module), name)
    ceiling = max(_SHARED_CEILING.get(key, 0), int(shared))
    _SHARED_CEILING[key] = ceiling
    if name.startswith(("mira_matrix_scan_p1i_", "mira_matrix_scan_p3i_")):
        D.prefer_shared(module[name], module["__ctx__"], ceiling, percent=-1)
    else:
        D.prefer_shared(module[name], module["__ctx__"], ceiling)


def launch(module, name, index, grid, block, pointers, scalars, shared=0):
    _prefer_shared(module, name, shared)
    ok, why = D.launch_dims(
        module[name], module["__ctx__"], index, grid, block,
        pointers, scalars, shared=shared)
    if not ok:
        raise RuntimeError(f"{name} {why}")


def _scan_name(phase, order, memory, lanes, group, suffix=None):
    tail = f"_{suffix}" if suffix is not None else ""
    return f"mira_matrix_scan_{phase}_k{order}_m{memory}_l{lanes}_g{group}{tail}"


# THE MAPPING IS A THUMB RULE, NOT A SEARCH. Planning used to ask the driver about every
# candidate -- local spill, resident blocks, the shared-memory opt-in -- once per shape, and the
# same questions were asked again at every new shape. The mapping is arithmetic and is solved here
# once, in closed form, so no launch queries the device and no plan iterates over candidates.
#
# The scan takes lanes = max(K, d_m / 4), so a lane carries four memory slots and the channel group
# fills the block out to TARGET_THREADS: 4 x 128 at d_m 16, 8 x 64 at 32, 16 x 32 at 64. The floor
# at K is p1's, whose phi is K basis vectors propagated on K lanes, and it BINDS EXACTLY ONCE in
# the compiled set -- at K = 8, d_m = 16, where max(8, 4) = 8 gives 8 x 64 instead of 4 x 128. For
# K = 1, 2 and 4 the floor is slack at all three memory dimensions, so those nine shapes keep the
# lane and group counts above unchanged.
#
# WIDENING THE LANES IS NOT THE BETTER SIDE OF THE TRADE, and this comment said the opposite until
# 2026-09-03. Every lane of a channel derives that channel's own (r cos, r sin) pair from alpha and
# beta -- two reciprocals and eleven further operations per stage -- because sharing it costs more
# than repeating it: the shuffle broadcast that used to do so was removed as measured-worse, and
# staging it in shared removes no global byte and measured worse again. So the coefficient
# arithmetic is replicated once per lane, and the lane count is what bounds that replication. Four
# lanes replicate it four times; one lane per memory slot would replicate it sixteen times and add
# more FP32 work than dropping p1 entirely would remove.
#
# The backward readout b3 takes the same lane count with the channel group and the token chunk
# that keep its shared block under 48 KiB -- the budget a block gets WITHOUT the opt-in, so the
# request can never be refused. That block is the SIX-TERM sum _backward_shared computes,
#     ((segments-1) G d_m K + S G d_m K + S G (K+1) + chunk d_m (K+3) + G d_m (K+2)
#      + chunk G K span) * 4 + read_cache
# at S = BACKWARD_SEGMENT, segments = ceil(chunk / S), span = STAGE_SPAN[mode] = 4 for the
# canonical mode 3, and read_cache = chunk (K+1) d_m element_size, which is charged only at
# K = 4 in the 2-byte dtypes. NO SINGLE TERM DECIDES THE ROW: this comment used to justify the
# rows with chunk * d_m * (K + 3) alone and quoted 22,528 B for (64, 8), and that arithmetic is
# what let three K = 8 rows ship over the ceiling -- the honest six-term figures for those rows
# were 71,936 B at (16, 8), 54,400 B at (32, 8) and 63,040 B at (64, 8), every one of them past
# 49,152. The rows below are the solution of the full sum, each shown at the worse of
# element_size 4 and 2 and every one inside the budget:
#     d_m=16 K=1  chunk 32  16 x 16  23,552 B      d_m=16 K=2  chunk 32  16 x 16  38,400 B
#     d_m=16 K=4  chunk 16  16 x 16  38,912 B      d_m=16 K=8  chunk 16  16 x  8  39,168 B
#     d_m=32 K=2  chunk 32  32 x  8  39,680 B      d_m=32 K=4  chunk 16  32 x  8  39,168 B
#     d_m=64 K=2  chunk 16  32 x  4  29,056 B      d_m=64 K=4  chunk  8  32 x  4  28,288 B
# (d_m, K) -> (chunk, lanes, group) for b3. THE SECOND TERM OF THAT SUM IS GONE AND THE GROUPS
# DOUBLED WITH IT. S G d_m K was the segment's replayed state, and every element of it was
# written and read back by the same thread -- same channel, same lane, same memory slot -- so it
# was 32,768 B of shared memory at (16, 4) carrying nothing any other thread could see. It is a
# register array in backward_phase_three now, S (d_m/L) K per thread, which is 32 at (16, 4)
# against the 256 a thread may hold at 256 threads and one block per SM. Freeing it took (16, 4)
# from 71,680 B at group 16 -- which is why that variant was unreachable and the row sat at group
# 8 -- to 38,912 B, so b3 runs at 256 threads instead of 128 and 8 warps per SM instead of 4.
# mira_v2: (d_m, K) -> (lanes, group) of b3. THE CHUNK COLUMN IS GONE: _fine_chunk derives the
# chunk from the shared memory of the GPU in use, and the group is the widest one it allows.
BACKWARD_MAP = {
    (8, 1): (8, 32), (8, 2): (8, 16), (8, 4): (8, 8), (8, 8): (8, 4),
    (16, 1): (16, 16), (16, 2): (16, 16), (16, 4): (16, 16), (16, 8): (16, 8),
                       (32, 2): (32, 8),  (32, 4): (32, 8),
                       (64, 2): (32, 16), (64, 4): (32, 4),
}


def _backward_entry(order, memory):
    entry = BACKWARD_MAP.get((memory, order))
    if entry is None:
        raise RuntimeError(
            f"mira-dsp has no backward mapping for order={order}, d_m={memory}; add the row to "
            f"BACKWARD_MAP rather than searching for one at run time")
    return entry


def _scan_plan(module, index, order, memory, suffix, batch, chunks, channels):
    """The widest lane count the memory axis allows, with the channel group filling the block."""
    # THE ORDER IS AN INPUT TO THE MAPPING AND MUST NOT BE DISCARDED. Deleting it here is what
    # hid the K = 8 lane deficit: the plan returned d_m / 4 lanes for every order, so at d_m = 16
    # a K = 8 launch asked for the l4_g128 kernels whose phi is eight columns wide on four lanes.
    del module, index, suffix, batch, chunks, channels
    return _scan_variants(memory, order)[-1]


def _inference_scan_plan(order, memory):
    return _inference_scan_variants(memory, order)[-1]


def _inference_replay_plan(order, memory):
    return _inference_replay_variants(memory, order)[-1]


def _forward_scan_name(phase, order, memory, lanes, group, suffix, keep):
    if not keep and order == 2 and (memory, lanes, group) in (
            (16, 2, 192), (16, 1, 128), (64, 8, 32), (64, 4, 32)):
        phase = f"{phase}i"
    return _scan_name(phase, order, memory, lanes, group, suffix)


def _kernel_resources(module, index, name, threads, dynamic_shared=0, blocks=None):
    function = module[name]
    context = module["__ctx__"]
    multiprocessors = D.sm_count(index)
    resident_blocks = D.resident_blocks(
        function, context, index, threads, dynamic_shared)
    blocks_per_multiprocessor = (
        resident_blocks // multiprocessors if resident_blocks and multiprocessors else 0)
    result = {
        "function": name,
        "threads_per_block": threads,
        "warps_per_block": threads // WARP_WIDTH,
        "dynamic_shared_memory_bytes": dynamic_shared,
        "static_shared_memory_bytes": D.function_attr(
            function, context, D.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES),
        "local_memory_bytes_per_thread": D.function_attr(
            function, context, D.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),
        "registers_per_thread": D.function_attr(
            function, context, D.CU_FUNC_ATTRIBUTE_NUM_REGS),
        "binary_compute_capability": D.function_attr(
            function, context, D.CU_FUNC_ATTRIBUTE_BINARY_VERSION),
        "preferred_shared_memory_carveout_percent": D.function_attr(
            function, context, D.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT),
        "maximum_dynamic_shared_memory_bytes": D.function_attr(
            function, context, D.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES),
        "resident_blocks_per_multiprocessor": blocks_per_multiprocessor,
        "resident_warps_per_multiprocessor": (
            blocks_per_multiprocessor * threads // WARP_WIDTH),
        "resident_blocks_device": resident_blocks,
    }
    if blocks is not None:
        result["grid_blocks"] = blocks
        result["resident_grid_waves"] = (
            blocks / resident_blocks if resident_blocks else None)
    return result


def _inference_chunk(module, index, order, memory, suffix, batch, tokens, channels):
    """The inference token chunk: _forward_chunk with the inference kernels and 16-token alignment."""
    return _forward_chunk(module, index, order, memory, suffix, batch, tokens, channels,
                          keep=False, fine=None)


def _composition_details(module, index, batch, chunks, channels, order, memory,
                         inference=False):
    plan = _tree_plan(module, index, batch, chunks, channels, order, memory)
    return {
        "method": "tree",
        "chunks": chunks,
        "memory_columns_per_thread": plan["columns"],
        "slices": plan["slices"],
        "units": plan["units"],
        "workers": plan["workers"],
        "chunks_per_worker": plan["per_worker"],
        "depth_compositions": plan["depth"],
        "inference_reads_last_chunk": not inference,
        "resources": _kernel_resources(
            module, index, _tree_name(order, memory), TREE_THREADS, blocks=plan["blocks"]),
    }


def mapping_details(index, order, memory, suffix, batch, tokens, channels):
    module = D.module_for(CUBIN, index, FNS)
    if module is None:
        return None
    chunk = _inference_chunk(
        module, index, order, memory, suffix, batch, tokens, channels)
    chunks = (tokens + chunk - 1) // chunk
    lanes, group = _inference_scan_plan(order, memory)
    groups = (channels + group - 1) // group
    blocks = batch * chunks * groups
    threads = lanes * group
    stage_tile = min(ALPHA_TILE, chunk)
    write_shared = 2 * stage_tile * memory * 4
    read_shared = 2 * stage_tile * (order + 1) * memory * 4
    planes = batch * chunks * channels
    p1_resources = _kernel_resources(
        module, index, _forward_scan_name(
            "p1", order, memory, lanes, group, suffix, False),
        threads, write_shared, blocks)
    replay_lanes, replay_group = _inference_replay_plan(order, memory)
    replay_groups = (channels + replay_group - 1) // replay_group
    p3_resources = _kernel_resources(
        module, index, _forward_scan_name(
            "p3", order, memory, replay_lanes, replay_group, suffix, False),
        replay_lanes * replay_group, write_shared + read_shared,
        batch * chunks * replay_groups)
    split = _split_plan(module, index, order, memory, suffix, batch, tokens, channels,
                        keep=False, fine=None)
    resident_blocks = min((
        value for value in (
            p1_resources["resident_blocks_device"],
            p3_resources["resident_blocks_device"])
        if value), default=0)
    return {
        "batch": batch,
        "tokens": tokens,
        "channels": channels,
        "lattice_order": order,
        "memory_size": memory,
        "dtype_suffix": suffix,
        "stage_mode": CANONICAL_STAGE,
        "token_chunk": chunk,
        "chunks": chunks,
        "chunk_alignment": ALPHA_TILE,
        "split_model": split,
        "channel_groups": groups,
        "planes": planes,
        "phi_elements_fp32": planes * order * order,
        "phi_bytes": planes * order * order * 4,
        "carry_elements_fp32": planes * memory * order,
        "carry_bytes": planes * memory * order * 4,
        "boundary_storage_bytes": (
            planes * order * order * 4 + planes * memory * order * 4),
        "optimization": {
            "resident_blocks_device": resident_blocks,
            "blocks_per_chunk": batch * groups,
            "selected_chunks": chunks,
            "selected_grid_blocks": blocks,
            "selected_resident_grid_waves": (
                (blocks + resident_blocks - 1) // resident_blocks
                if resident_blocks else None),
        },
        "scan": {
            "lanes_per_channel": lanes,
            "memory_values_per_lane": memory // lanes,
            "channels_per_block": group,
            "p1": p1_resources,
            "p3": p3_resources,
        },
        "carry_composition": _composition_details(
            module, index, batch, chunks, channels, order, memory, inference=True),
    }


def mapping_inventory(index, suffix):
    module = D.module_for(CUBIN, index, FNS)
    if module is None:
        return None
    entries = []
    for order, memory in _built_pairs():
        lanes, group = _scan_variants(memory, order)[-1]
        compose_lanes, compose_group = _compose_variants(memory, order)[-1]
        backward_chunk, backward_lanes, backward_group, backward_shared = _fine_chunk_device(
            index, order, memory, 4 if suffix == "f32" else 2, module)
        write_shared = 2 * ALPHA_TILE * memory * 4
        read_shared = 2 * ALPHA_TILE * (order + 1) * memory * 4
        backward_read_shared = (
            ALPHA_TILE * (order + 1) * memory * (4 if suffix == "f32" else 2))
        scan_threads = lanes * group
        compose_threads = compose_lanes * compose_group
        backward_threads = backward_lanes * backward_group
        entries.append({
            "lattice_order": order,
            "memory_size": memory,
            "training_token_chunk": backward_chunk,
            "forward_scan": {
                "lanes_per_channel": lanes,
                "channels_per_block": group,
                "threads_per_block": lanes * group,
            },
            "carry_composition": {
                "method": "tree",
                "memory_columns_per_thread": _tree_columns(order, memory),
                "threads_per_block": TREE_THREADS,
            },
            "backward_replay": {
                "lanes_per_channel": backward_lanes,
                "channels_per_block": backward_group,
                "threads_per_block": backward_lanes * backward_group,
                "dynamic_shared_memory_bytes": backward_shared,
            },
            "resources": {
                "p1": _kernel_resources(
                    module, index, _scan_name(
                        "p1", order, memory, *_phase_one_variants(memory, order)[-1], suffix),
                    _phase_one_variants(memory, order)[-1][0]
                    * _phase_one_variants(memory, order)[-1][1], write_shared),
                "tree": _kernel_resources(
                    module, index, _tree_name(order, memory), TREE_THREADS),
                "p3": _kernel_resources(
                    module, index, _scan_name(
                        "p3", order, memory, lanes, group, suffix),
                    scan_threads, write_shared + read_shared),
                "b1": _kernel_resources(
                    module, index, _scan_name(
                        "b1", order, memory, lanes, group, suffix),
                    scan_threads, backward_read_shared),
                "b3": _kernel_resources(
                    module, index, _scan_name(
                        "b3", order, memory, backward_lanes, backward_group, suffix),
                    backward_threads, backward_shared),
                "single_token": _kernel_resources(
                    module, index, f"mira_matrix_step_k{order}_m{memory}_{suffix}",
                    COMPOSE_THREADS, (memory + (order + 1) * memory) * 4),
            },
        })
    return {
        "cubin": module["__path__"],
        "compiled_functions": list(FNS),
        "compiled_pairs": entries,
        "constants": {
            "warp_width": WARP_WIDTH,
            "scan_threads": TARGET_THREADS,
            "composition_threads": COMPOSE_THREADS,
            "backward_threads": BACKWARD_THREADS,
            "backward_segment": dict(BACKWARD_SEGMENT),
            "channel_tile": CHANNEL_TILE,
            "reduction_threads": REDUCTION_THREADS,
            "cooperative_token_tile": ALPHA_TILE,
            "canonical_stage": CANONICAL_STAGE,
            "tree_threads": TREE_THREADS,
            "tree_max_workers": TREE_MAX_WORKERS,
            "split_margin": SPLIT_MARGIN,
            "launch_bounds_blocks_per_multiprocessor": dict(LAUNCH_BOUNDS_BLOCKS),
            "hardware": _hardware(index),
        },
        "common_resources": {
            "backward_reduction": _kernel_resources(
                module, index, f"mira_matrix_scan_b4_k4_{suffix}",
                REDUCTION_THREADS),
        },
    }


def _backward_shared(order, memory, group, chunk, stage_mode, element_size):
    segment = BACKWARD_SEGMENT[memory]
    segments = (chunk + segment - 1) // segment
    read_cache = chunk * (order + 1) * memory * element_size if (
        order == 4 and element_size == 2) else 0
    return (
        (segments - 1) * group * memory * order
        + segment * group * (order + 1)
        + chunk * memory * (order + 3)
        + group * memory * (order + 2) * (2 if order == 2 and memory == 16 else 1)
        + chunk * group * order * STAGE_SPAN[stage_mode]
    ) * 4 + read_cache


# ------------------------------------------------------------------------------------------------
# mira_v2: THE HARDWARE FACTS AND THE CHUNK RULES
#
# EVERY NUMBER BELOW IS A CLOSED FORM OF THE SHAPE AND OF FACTS THE DRIVER REPORTS ABOUT THIS GPU.
# Nothing is searched, timed or tuned (user/Rules.md: trainers and profilers never search). The
# facts are read once per device and the occupancy once per (device, kernel, block, shared size),
# and cached; no launch asks the driver anything new for a shape it has seen.
#
# THE UNIT OF PARALLELISM IS THE RESIDENT BLOCK, R = SMs x blocks per SM of the compiled kernel, as
# in FlashAttention (flash_api.cpp: "multiply number of SMs by 2 ... 128 threads per block"), except
# that the factor is the driver's answer for this kernel, not a constant.
# ------------------------------------------------------------------------------------------------

CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81
CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111
_HARDWARE = {}
_OCCUPANCY = {}


def _hardware(index):
    """SM count and shared memory of device `index` (bytes), read once per process.

    A device the driver will not describe gets 1 SM and the 48 KiB every block may use without the
    opt-in, which only ever makes the rules below choose fewer chunks and a shorter backward chunk."""
    facts = _HARDWARE.get(index)
    if facts is not None:
        return dict(facts)
    optin = D.max_optin_shared(index)
    facts = {
        "sms": int(D.sm_count(index) or 0),
        "shared_optin_per_block": int(optin),
        "shared_per_multiprocessor": int(D.device_attr(
            index, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, 0) or optin),
        "shared_reserved_per_block": max(0, int(D.device_attr(
            index, CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK, 0) or 0)),
    }
    if facts["sms"]:
        _HARDWARE[index] = dict(facts)
    facts["sms"] = max(1, facts["sms"])
    return facts


def _phase_of(name):
    if name.startswith("mira_tree_scan_"):
        return "tree"
    for phase in ("p1i", "p3i", "p1", "p3", "b3"):
        if f"_scan_{phase}_" in name:
            return phase
    raise ValueError(f"no occupancy floor for {name}")


def _blocks_per_sm(module, index, name, threads, shared):
    """Resident blocks per SM of kernel `name` at `threads` threads and `shared` dynamic bytes.

    The driver's cuOccupancyMaxActiveBlocksPerMultiprocessor for the compiled function, asked after
    the carveout the launch will use has been set (the answer depends on it); the __launch_bounds__
    minimum (LAUNCH_BOUNDS_BLOCKS) when there is no module or the driver will not say."""
    floor = LAUNCH_BOUNDS_BLOCKS[_phase_of(name)]
    if module is None or name not in module:
        return floor
    key = (index, name, int(threads), int(shared))
    cached = _OCCUPANCY.get(key)
    if cached:
        return cached
    function = module[name]
    _prefer_shared(module, name, shared)
    value = D._blocks_per_sm(function, module["__ctx__"], threads, shared)
    if value > 0:
        _OCCUPANCY[key] = value
        return value
    return floor


def _split_constants(order, memory, element_size, lanes_p1, lanes_p3):
    """(c0, rho): the per-block fixed cost in token-times and the cost of p1 relative to p3.

    c0 = the first 16-token staging tile, which the double buffer does not overlap, plus the
    boundary bytes of a chunk (fp32 Phi and carry written by p1, read and written by the
    composition, read by p3: 4 (3K^2 + 5K d_m)) over the bytes p1 and p3 stream per token
    (e (5K + 5)). rho = the larger of p1's bytes over p3's, (2K+1)/(3K+4), and p1's FP32 instructions
    over p3's -- memory-bound GPUs follow the first, instruction-bound ones the second."""
    K, M, e = order, memory, element_size
    c0 = ALPHA_TILE + 4.0 * (3 * K * K + 5 * K * M) / (e * (5 * K + 5))
    a_p3 = 4 * K * M + 2 * (K + 1) * M + 13 * K * lanes_p3 + 10 * lanes_p3
    a_p1 = 4 * K * M + 4 * K * K + 13 * K * lanes_p1 + M
    rho = max((2 * K + 1) / (3 * K + 4), a_p1 / a_p3)
    return c0, rho


def _forward_kernels(order, memory, suffix, keep):
    """((p1 name, lanes, group, shared), (p3 name, lanes, group, shared)) the forward launches."""
    if keep:
        lanes_p1, group_p1 = _phase_one_variants(memory, order)[-1]
        lanes_p3, group_p3 = _scan_variants(memory, order)[-1]
    else:
        lanes_p1, group_p1 = _inference_scan_plan(order, memory)
        lanes_p3, group_p3 = _inference_replay_plan(order, memory)
    write_shared = ALPHA_TILE * memory * 4
    read_shared = ALPHA_TILE * (order + 1) * memory * 4
    return ((_forward_scan_name("p1", order, memory, lanes_p1, group_p1, suffix, keep),
             lanes_p1, group_p1, 2 * write_shared),
            (_forward_scan_name("p3", order, memory, lanes_p3, group_p3, suffix, keep),
             lanes_p3, group_p3, 2 * (write_shared + read_shared)))


def _split_plan(module, index, order, memory, suffix, batch, tokens, channels, keep, fine):
    """How the forward cuts the token axis: the closed form of doc rule F (mira_v2).

        x    = B ceil(d_e / G_p3) / R_p3                      waves of the unsplit p3 grid
        W    = max(ceil(x), floor(x sqrt(T / c0)))             whole waves
        n    = min(floor(W R_p3 / (B ceil(d_e / G_p3))), floor(sqrt T), ceil(T / a))
        c    = min(T, a ceil(T / (n a))),   n = ceil(T / c)    a = 16 (inference) | fine (training)
        t(n) = ceil(n N3 / R3) (c + c0) + rho ceil(n N1 / R1) c
        t(1) = ceil(x) T  (+ rho ceil(N1 / R1) T in training, where p1 runs at one chunk too)
        one chunk unless t(1) > SPLIT_MARGIN t(n)

    The grid lands on W whole waves of resident blocks: FlashAttention's one-wave split
    (nsplits = SMs / (batch heads)) when the batch does not fill the GPU, and more waves only when
    the rounding loss of the last one (about x^2 T / W) exceeds their boundary cost (W c0). At most
    sqrt(T) chunks, because the chunks are joined one after another (depth n) and that chain must
    not outgrow a chunk (length T / n)."""
    align = int(fine) if keep else ALPHA_TILE
    facts = _hardware(index)
    sms = facts["sms"]
    (p1, lanes_p1, group_p1, shared_p1), (p3, lanes_p3, group_p3, shared_p3) = \
        _forward_kernels(order, memory, suffix, keep)
    blocks_p3 = _blocks_per_sm(module, index, p3, lanes_p3 * group_p3, shared_p3)
    blocks_p1 = _blocks_per_sm(module, index, p1, lanes_p1 * group_p1, shared_p1)
    N3 = batch * ((channels + group_p3 - 1) // group_p3)
    N1 = batch * ((channels + group_p1 - 1) // group_p1)
    R3, R1 = sms * blocks_p3, sms * blocks_p1
    c0, rho = _split_constants(order, memory, 4 if suffix == "f32" else 2, lanes_p1, lanes_p3)
    x = N3 / R3
    plan = {"resident_p3": R3, "resident_p1": R1, "blocks_p3": N3, "blocks_p1": N1,
            "waves_unsplit": x, "c0": c0, "rho": rho, "alignment": align}
    t1 = math.ceil(x) * tokens + (rho * math.ceil(N1 / R1) * tokens if keep else 0.0)
    if tokens <= align:
        plan.update(waves=1, chunks=1, chunk=tokens, split=False, time_split=t1, time_one=t1)
        return plan
    waves = max(math.ceil(x), math.floor(x * math.sqrt(tokens / c0)))
    chunks = max(1, min(waves * R3 // N3, math.isqrt(tokens), -(-tokens // align)))
    chunk = min(tokens, align * -(-tokens // (chunks * align)))
    chunks = -(-tokens // chunk)
    tn = (math.ceil(chunks * N3 / R3) * (chunk + c0) + rho * math.ceil(chunks * N1 / R1) * chunk
          if chunks > 1 else t1)
    split = chunks > 1 and t1 > SPLIT_MARGIN * tn
    plan.update(waves=waves, chunks=chunks if split else 1, chunk=chunk if split else tokens,
                split=split, time_split=tn, time_one=t1)
    return plan


def _forward_chunk(module, index, order, memory, suffix, batch, tokens, channels, keep, fine):
    """The forward token chunk (training: a multiple of `fine`, or the whole sequence)."""
    return _split_plan(module, index, order, memory, suffix, batch, tokens, channels,
                       keep, fine)["chunk"]


def _b3_budget(module, index, order, memory, suffix, stage_mode):
    """(lanes, group, f_smem, sigma): the b3 variant and the longest chunk its shared block allows.

    b3's shared block is exactly affine in the chunk for multiples of the 8-token segment,
    A + B f (_backward_shared). The budget is what this GPU gives a block when it is to hold
    beta = min(2, blocks its registers allow) blocks per SM,
        sigma = min(opt-in per block, shared per SM / beta - reserved per block),
    and f_smem is the largest multiple of the segment inside it. The BACKWARD_MAP group is tried
    first, then the narrower compiled groups when it leaves fewer than two segments; when even the
    narrowest leaves less than one at two blocks per SM, the budget is one block per SM."""
    facts = _hardware(index)
    element_size = 4 if suffix == "f32" else 2
    lanes, widest = _backward_entry(order, memory)
    groups = [widest] + sorted({group for variant_lanes, group in _b3_variants(order, memory)
                                if variant_lanes == lanes and group < widest}, reverse=True)
    segment = BACKWARD_SEGMENT[memory]
    for target in (2, 1):
        for group in groups:
            name = _scan_name("b3", order, memory, lanes, group, suffix)
            beta = max(1, min(target, _blocks_per_sm(module, index, name, lanes * group, 0)))
            sigma = min(facts["shared_optin_per_block"],
                        facts["shared_per_multiprocessor"] // beta
                        - facts["shared_reserved_per_block"])
            one = _backward_shared(order, memory, group, segment, stage_mode, element_size)
            two = _backward_shared(order, memory, group, 2 * segment, stage_mode, element_size)
            slope = (two - one) / segment
            base = one - segment * slope
            f_smem = segment * int((sigma - base) // (segment * slope))
            if f_smem < 2 * segment and group != groups[-1]:
                continue
            if f_smem < segment:
                continue
            return lanes, group, f_smem, sigma
    raise RuntimeError(
        f"mira_v2: no b3 variant for order={order}, d_m={memory} fits this GPU's shared memory "
        f"({facts['shared_optin_per_block']} B per block)")


def _pow2floor(value):
    return 1 << (max(1, int(value)).bit_length() - 1)


def _fine_chunk(module, index, order, memory, suffix, batch, tokens, channels, stage_mode):
    """(f, lanes, group, shared): the backward fine chunk of doc rule B (mira_v2).

        f_par = 8 floor(B T ceil(d_e / C_tile) / (8 R_b3))       b3 keeps at least one wave
        f     = max(8, 2^floor(log2 min(f_smem, f_par)))

    A power of two so a power-of-two T aligns; f_smem from _b3_budget."""
    lanes, group, f_smem, _sigma = _b3_budget(module, index, order, memory, suffix, stage_mode)
    element_size = 4 if suffix == "f32" else 2
    segment = BACKWARD_SEGMENT[memory]
    name = _scan_name("b3", order, memory, lanes, group, suffix)
    resident = _hardware(index)["sms"] * _blocks_per_sm(
        module, index, name, lanes * group,
        _backward_shared(order, memory, group, f_smem, stage_mode, element_size))
    tile = CHANNEL_TILES.get((order, memory), CHANNEL_TILE)
    f_par = segment * ((batch * tokens * ((channels + tile - 1) // tile)) // (segment * resident))
    f = max(segment, _pow2floor(min(f_smem, max(f_par, segment))))
    return f, lanes, group, _backward_shared(order, memory, group, f, stage_mode, element_size)


def _fine_chunk_device(index, order, memory, element_size, module=None, stage_mode=CANONICAL_STAGE):
    """(f, lanes, group, shared) at a shape large enough that only shared memory binds: the
    longest backward chunk this GPU allows, for the report and the mapping inventory."""
    suffix = "f32" if element_size == 4 else "bf16"
    lanes, group, f_smem, _sigma = _b3_budget(module, index, order, memory, suffix, stage_mode)
    f = _pow2floor(f_smem)
    return f, lanes, group, _backward_shared(order, memory, group, f, stage_mode, element_size)


def _backward_plan(module, index, order, memory, suffix, batch, tokens, channels, chunk,
                   stage_mode=CANONICAL_STAGE):
    """(lanes, group, shared) of b3 for the fine chunk the forward used, by the same rule."""
    f, lanes, group, _shared = _fine_chunk(
        module, index, order, memory, suffix, batch, tokens, channels, stage_mode)
    if min(f, tokens) != chunk:
        raise RuntimeError(f"mira_v2: backward chunk {min(f, tokens)} differs from the forward's {chunk}")
    return lanes, group, _backward_shared(
        order, memory, group, chunk, stage_mode, 4 if suffix == "f32" else 2)


def _tree_name(order, memory):
    return f"mira_tree_scan_k{order}_m{memory}"


def _tree_columns(order, memory):
    """V, the memory columns one tree thread carries: K V <= 8 (mira_iir.cu instantiations)."""
    return max(1, min(memory, 8 // order))


def _pow2ceil(value):
    return 1 << max(0, int(value) - 1).bit_length()


def _tree_plan(module, index, batch, chunks, channels, order, memory):
    """How the tree scan covers `chunks` chunks of batch x channels streams (closed form, no search).

        units   = B d_e (d_m / V)                         independent (sequence, channel, slice)
        R       = SMs x resident tree blocks per SM x 256  threads the GPU holds at once
        workers = clamp(2^ceil(log2(R / units)), 1, min(2^ceil(log2 chunks), 256))
        r       = ceil(chunks / workers)                  chunks one worker folds and replays

    Enough workers per unit to fill the GPU once, never more than the chunks need (Brent): depth
    2 r + log2(workers) + workers / 32 compositions instead of v2's chunks."""
    columns = _tree_columns(order, memory)
    slices = memory // columns
    units = batch * channels * slices
    name = _tree_name(order, memory)
    resident = (_hardware(index)["sms"] * _blocks_per_sm(module, index, name, TREE_THREADS, 0)
                * TREE_THREADS)
    workers = _pow2ceil(-(-resident // max(1, units)))
    workers = max(1, min(workers, _pow2ceil(max(1, chunks)), TREE_MAX_WORKERS))
    per_worker = -(-chunks // workers)
    depth = (per_worker if workers == 1 else
             2 * per_worker + (workers.bit_length() - 1) + max(0, workers // 32 - 1))
    blocks = -(-units * workers // TREE_THREADS)
    return {"columns": columns, "slices": slices, "units": units, "workers": workers,
            "per_worker": per_worker, "depth": depth, "blocks": blocks}


def _compose_boundaries(module, index, phi, carry, batch, chunks, channels, order, memory,
                        initial=None, reverse=False, inference=False):
    """The state entering every chunk, written over `carry` in place: one tree-scan launch."""
    plan = _tree_plan(module, index, batch, chunks, channels, order, memory)
    launch(module, _tree_name(order, memory), index, plan["blocks"], TREE_THREADS,
           [phi.data_ptr(), carry.data_ptr(),
            0 if (initial is None or reverse) else initial.data_ptr()],
           [batch, chunks, channels, plan["workers"], int(reverse), int(inference)])
    return carry


def make_function():
    import torch

    class MatrixLatticeIIR(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx, x, raw_alpha, raw_beta, raw_rho, gamma, memory_write, memory_read,
            order, memory, chunk, suffix, stage_mode, row_stride,
            gamma_row_stride, alpha_row_stride, beta_row_stride, rho_row_stride,
            read_row_stride, keep,
            packed_input=None, packed_coefficients=None, valid=None, state=None, return_state=False,
        ):
            index = x.device.index
            module = D.module_for(CUBIN, index, FNS)
            if module is None:
                raise RuntimeError("mira-dsp cubin is unavailable")
            # RAW_BETA IS MODE 3'S ALONE. The other three stages never dereference it, so a null
            # pointer is what they are given rather than a tensor they would ignore.
            beta_pointer = 0 if raw_beta is None else raw_beta.data_ptr()
            batch, tokens, channels = x.shape
            fine_chunk = chunk
            if keep:
                # THE COARSE CHUNK IS _forward_chunk's, A MULTIPLE OF THE FINE CHUNK (p1 writes the
                # fine transitions at fine edges) or the whole sequence.
                chunk = _forward_chunk(module, index, order, memory, suffix, batch, tokens,
                                       channels, keep=True, fine=fine_chunk)
            chunks = (tokens + chunk - 1) // chunk
            planes = batch * chunks * channels
            lanes, channel_group = (
                _scan_plan(module, index, order, memory, suffix, batch, chunks, channels)
                if keep else _inference_scan_plan(order, memory))
            channel_groups = (channels + channel_group - 1) // channel_group
            grouped_planes = batch * chunks * channel_groups
            threads = lanes * channel_group
            stage_tile = min(ALPHA_TILE, chunk)
            write_shared = stage_tile * memory * 4
            read_shared = stage_tile * (order + 1) * memory * 4
            output = torch.empty(
                batch, tokens, channels, device=x.device, dtype=x.dtype)
            # THE SUM OVER k, KEPT ONLY WHEN SiLU IS ON IT AND A GRADIENT IS WANTED. b1 walks the
            # reverse scan without replaying the levels, so it cannot form the sum and reads it
            # from here; b3 replays them and forms it itself. Without the SiLU nothing reads it and
            # p3 is given a null pointer.
            # ALWAYS NULL SINCE readout_silu WAS DROPPED (2026-09-14). It existed to hold the
            # pre-activation readout for the backward's silu_grad; with no activation on the
            # readout there is nothing to hold. The kernel's `if (readout_sum)` guards keep it
            # null-safe, so p3's per-token store to it never fires either.
            readout_sum = None
            direct = tokens == 1 and not keep
            # ONE CHUNK AND NO RETAINED GRAPH: p1 AND THE COMPOSITION ARE A NO-OP ROUND TRIP.
            # At chunks == 1 every p1 block returns at its own `chunk_index + 1 == chunks`
            # guard, and p2's only act is to store the chunk's OPENING state --
            # `initial` when one was supplied, zeros otherwise -- into `checkpoint`, which p3
            # reads straight back. p3 reads `initial` by the SAME index expression when
            # `checkpoint` is null and keeps its own zero initialiser when both are, so a null
            # checkpoint is bit-identical to what p2 would have written.
            #
            # At batch 554, context 2,048, d_e 1,536, d_m 16, K=2 that is 850,944 planes x 32
            # float32: 108.9 MB written and 108.9 MB read, 192 of the 65,216 bytes a token the
            # mixer moves, two launches (4,432 blocks that return, and 53,184 of 64 threads), and
            # 122.5 MB the two buffers ask the allocator for.
            #
            # keep is excluded because p1's fine_phi and p2's composed carry are both backward
            # inputs, and chunks == 1 is reachable with keep at tokens <= 32. A supplied state is
            # excluded as well: p3's `initial` branch reproduces it exactly, but keeping the
            # composition there costs nothing and the profiled path passes none.
            single = chunks == 1 and not keep
            phi = None if direct or single else torch.empty(
                planes, order, order,
                device=x.device, dtype=torch.float32)
            carry = None if direct or single else torch.empty(
                planes, memory, order, device=x.device, dtype=torch.float32)
            fine_chunks = (tokens + fine_chunk - 1) // fine_chunk
            checkpoint = carry if keep and chunk == fine_chunk else (torch.empty(batch * fine_chunks * channels, memory, order,
                                      device=x.device, dtype=torch.float32) if keep else None)
            fine_phi = phi if keep and chunk == fine_chunk else (torch.empty(batch * fine_chunks * channels, order, order,
                                   device=x.device, dtype=torch.float32) if keep else None)
            final = (torch.empty(batch, order, channels, memory,
                                 device=x.device, dtype=torch.float32) if return_state else None)
            valid_pointer = 0 if valid is None else valid.data_ptr()
            x_pointer = x.data_ptr()
            alpha_pointer = raw_alpha.data_ptr()
            write_pointer = memory_write.data_ptr()
            carry_pointer = 0 if carry is None else carry.data_ptr()
            phi_pointer = 0 if phi is None else phi.data_ptr()
            if not direct and not single:
                p1_lanes, p1_group = (
                    _phase_one_variants(memory, order)[-1] if keep else (lanes, channel_group))
                launch(
                    module, _forward_scan_name(
                        "p1", order, memory, p1_lanes, p1_group, suffix, keep), index,
                    batch * chunks * ((channels + p1_group - 1) // p1_group),
                    p1_lanes * p1_group,
                    [x_pointer, alpha_pointer, beta_pointer,
                     write_pointer,
                     phi_pointer, carry_pointer,
                     0 if fine_phi is None or chunk == fine_chunk else fine_phi.data_ptr(),
                     valid_pointer],
                    [batch, tokens, channels, order, memory, chunk, stage_mode,
                     row_stride, alpha_row_stride, beta_row_stride, fine_chunk],
                    shared=2 * write_shared,
                )
                _compose_boundaries(module, index, phi, carry, batch, chunks, channels,
                                    order, memory, state, inference=not keep)
            del phi
            if not keep and not direct:
                lanes, channel_group = _inference_replay_plan(order, memory)
                threads = lanes * channel_group
                grouped_planes = batch * chunks * (
                    (channels + channel_group - 1) // channel_group)
            if direct:
                lanes, channel_group = _compose_variants(memory, order)[-1]
                threads = COMPOSE_THREADS
                grouped_planes = batch * ((channels + channel_group - 1) // channel_group)
            launch(
                module, (f"mira_matrix_step_k{order}_m{memory}_{suffix}" if direct else
                         _forward_scan_name(
                             "p3", order, memory, lanes, channel_group, suffix, keep)), index,
                grouped_planes,
                threads,
                [x_pointer, alpha_pointer, beta_pointer,
                 raw_rho.data_ptr(), gamma.data_ptr(),
                 write_pointer, memory_read.data_ptr(),
                 0 if direct else carry_pointer, output.data_ptr(),
                 0 if readout_sum is None else readout_sum.data_ptr(),
                 0 if checkpoint is None or chunk == fine_chunk else checkpoint.data_ptr(),
                 0 if state is None else state.data_ptr(),
                 0 if final is None else final.data_ptr(), valid_pointer],
                [batch, tokens, channels, order, memory, chunk, stage_mode,
                 row_stride, alpha_row_stride, beta_row_stride, rho_row_stride,
                 gamma_row_stride, read_row_stride, fine_chunk],
                shared=(write_shared + read_shared) * (1 if direct else 2),
            )
            ctx.save_for_backward(
                x, raw_alpha, raw_beta, raw_rho, gamma, memory_write, memory_read,
                checkpoint if keep else x.new_empty(0, dtype=torch.float32),
                readout_sum if readout_sum is not None
                else x.new_empty(0, dtype=torch.float32),
                fine_phi, valid, packed_input)
            ctx.meta = (
                batch, tokens, channels, order, memory, fine_chunk, suffix,
                stage_mode, row_stride, gamma_row_stride, alpha_row_stride,
                beta_row_stride, rho_row_stride, read_row_stride, lanes,
                channel_group)
            ctx.packed = packed_input is not None
            USED["forward"] += 1
            if return_state:
                ctx.mark_non_differentiable(final)
                return output, final
            return output

        @staticmethod
        def backward(ctx, grad_output, grad_final=None):
            (x, raw_alpha, raw_beta, raw_rho, gamma, memory_write, memory_read,
             checkpoint, readout_sum, fine_phi, valid, packed_input) = ctx.saved_tensors
            (batch, tokens, channels, order, memory, chunk, suffix,
             stage_mode, row_stride, gamma_row_stride, alpha_row_stride,
             beta_row_stride, rho_row_stride, read_row_stride, lanes,
             channel_group) = ctx.meta
            if checkpoint.numel() == 0:
                raise RuntimeError("mira-dsp backward requires forward checkpoints")
            beta_pointer = 0 if raw_beta is None else raw_beta.data_ptr()
            index = x.device.index
            module = D.module_for(CUBIN, index, FNS)
            grad_output = grad_output.contiguous()
            if grad_output.dtype != x.dtype:
                grad_output = grad_output.to(x.dtype)
            chunks = (tokens + chunk - 1) // chunk
            planes = batch * chunks * channels
            channel_groups = (channels + channel_group - 1) // channel_group
            grouped_planes = batch * chunks * channel_groups
            threads = lanes * channel_group
            read_shared = ALPHA_TILE * (order + 1) * memory * x.element_size()
            reverse_carry = torch.empty(
                planes, memory, order, device=x.device, dtype=torch.float32)
            boundary = reverse_carry
            alpha_pointer = raw_alpha.data_ptr()
            rho_pointer = raw_rho.data_ptr()
            gamma_pointer = gamma.data_ptr()
            read_pointer = memory_read.data_ptr()
            grad_output_pointer = grad_output.data_ptr()
            reverse_carry_pointer = reverse_carry.data_ptr()
            valid_pointer = 0 if valid is None else valid.data_ptr()
            launch(
                module, _scan_name(
                    "b1", order, memory, lanes, channel_group, suffix), index,
                grouped_planes,
                threads,
                [alpha_pointer, beta_pointer,
                 rho_pointer, gamma_pointer,
                 read_pointer, grad_output_pointer,
                 0 if readout_sum.numel() == 0 else readout_sum.data_ptr(),
                 valid_pointer, reverse_carry_pointer],
                [batch, tokens, channels, order, memory, chunk, stage_mode,
                 row_stride, alpha_row_stride, beta_row_stride, rho_row_stride,
                 gamma_row_stride, read_row_stride],
                shared=read_shared,
            )
            _compose_boundaries(module, index, fine_phi, reverse_carry, batch, chunks,
                                channels, order, memory, reverse=True)
            io = dict(device=x.device, dtype=x.dtype)
            if ctx.packed:
                grad_input = torch.empty(batch, tokens, row_stride, **io)
                grad_coefficients = torch.empty(batch, tokens, alpha_row_stride, **io)
                grad_x, grad_write = grad_input.split((channels, memory), dim=-1)
                grad_alpha, grad_beta, grad_rho, grad_read = grad_coefficients.split(
                    (channels * order, channels * order, channels * (order + 1),
                     (order + 1) * memory), dim=-1)
                grad_alpha = grad_alpha.view(batch, tokens, channels, order)
                grad_beta = grad_beta.view(batch, tokens, channels, order)
                grad_rho = grad_rho.view(batch, tokens, channels, order + 1)
                grad_read = grad_read.view(batch, tokens, order + 1, memory)
            else:
                grad_input = grad_coefficients = None
                grad_x = torch.empty(batch, tokens, channels, **io)
                grad_alpha = torch.empty(batch, tokens, channels, order, **io)
                grad_beta = torch.empty(batch, tokens, channels, order, **io)
                grad_rho = torch.empty(batch, tokens, channels, order + 1, **io)
                grad_write = torch.empty(batch, tokens, memory, **io)
                grad_read = torch.empty(batch, tokens, order + 1, memory, **io)
            grad_gamma = torch.empty(batch, tokens, channels, **io)
            channel_tile = CHANNEL_TILES.get((order, memory), CHANNEL_TILE)
            groups = (channels + channel_tile - 1) // channel_tile
            head_channels = (channels + groups - 1) // groups
            grad_memory_groups = torch.empty(
                groups, batch, tokens, order + 2, memory,
                device=x.device, dtype=torch.float32)
            backward_lanes, channel_group, shared = _backward_plan(
                module, index, order, memory, suffix, batch, tokens, channels, chunk,
                stage_mode)
            threads = backward_lanes * channel_group
            launch(
                module, _scan_name(
                    "b3", order, memory, backward_lanes, channel_group, suffix), index,
                batch * chunks * groups,
                threads,
                [x.data_ptr(), alpha_pointer, beta_pointer,
                 rho_pointer, gamma_pointer,
                 memory_write.data_ptr(), read_pointer,
                 checkpoint.data_ptr(), reverse_carry_pointer, grad_output_pointer,
                 grad_x.data_ptr(), grad_alpha.data_ptr(),
                 0 if grad_beta is None else grad_beta.data_ptr(),
                 grad_rho.data_ptr(),
                 grad_gamma.data_ptr(), grad_memory_groups.data_ptr(), valid_pointer],
                [batch, tokens, channels, order, memory, chunk, stage_mode,
                 groups, channel_group, row_stride, alpha_row_stride, beta_row_stride,
                 rho_row_stride, gamma_row_stride, read_row_stride, head_channels,
                 int(ctx.packed)],
                shared=shared,
            )
            values = batch * tokens * (order + 2) * memory
            launch(
                module, f"mira_matrix_scan_b4_k4_{suffix}", index,
                (values + REDUCTION_THREADS - 1) // REDUCTION_THREADS,
                REDUCTION_THREADS,
                [grad_memory_groups.data_ptr(), grad_write.data_ptr(), grad_read.data_ptr(),
                 0 if packed_input is None else packed_input.data_ptr() + channels * x.element_size()],
                [values, groups, order + 2, memory,
                 row_stride if ctx.packed else memory,
                 read_row_stride if ctx.packed else (order + 1) * memory],
            )
            USED["backward"] += 1
            # ONE None PER NON-TENSOR INPUT, AND THE COUNT IS ASSERTED RATHER THAN TRUSTED.
            #
            # forward takes eight differentiable tensors, then TWELVE scalars -- order, memory,
            # chunk, suffix, stage_mode, row_stride, gamma_row_stride, alpha_row_stride,
            # beta_row_stride, rho_row_stride, read_row_stride, keep -- then packed_input,
            # packed_coefficients, valid, state, return_state.  backward must return exactly one
            # gradient per input, in that order.
            #
            # It read `[None] * 13` while there were thirteen scalars.  Dropping readout_silu on
            # 2026-09-14 left twelve and the literal behind, so grad_input landed on
            # packed_coefficients and grad_coefficients on valid, and `make quicktest` raised
            #
            #     returned an invalid gradient at index 8 - got [2, 33, 139] but expected
            #     shape compatible with [2, 33, 540]
            #
            # -- a shape error several arguments away from the argument that moved.  The
            # assertion below is what makes the next signature change say so immediately.
            scalars = 12
            gradients = [
                grad_x, grad_alpha, grad_beta, grad_rho, grad_gamma, grad_write, grad_read,
                *([None] * scalars), grad_input, grad_coefficients, None, None, None]
            if ctx.packed:
                for i in (0, 1, 2, 3, 5, 6):
                    gradients[i] = None
            assert len(gradients) == len(ctx.needs_input_grad), (
                f"mira-dsp backward returns {len(gradients)} gradients for "
                f"{len(ctx.needs_input_grad)} forward inputs; the two signatures have drifted")
            return tuple(gradients[:len(ctx.needs_input_grad)])


    return MatrixLatticeIIR


def matrix_lattice_iir(
    x, raw_alpha, raw_rho, gamma, memory_write, memory_read, raw_beta=None,
    valid=None, state=None, return_state=False, packed=None, arch="cauchy_x_hyperbolic",
):
    global FUNCTION
    if arch not in ARCH_STAGE:
        raise ValueError(f"arch must be one of {tuple(ARCH_STAGE)}, not {arch!r}")
    try:
        import torch
    except Exception:
        return None
    if not (x.dim() == 3 and gamma.dim() == 3 and raw_alpha.dim() == 4
            and raw_rho.dim() == 4 and memory_write.dim() == 3
            and memory_read.dim() == 4):
        return None
    batch, tokens, channels = x.shape
    if not batch or not tokens or not channels:
        raise ValueError("mira-dsp requires nonempty batch, token and channel axes")
    order = int(raw_alpha.shape[-1])
    memory = int(memory_write.shape[-1])
    if order not in ORDERS or memory not in ORDER_MEMORIES[order]:
        return None
    if raw_alpha.shape != (batch, tokens, channels, order):
        return None
    if raw_rho.shape != (batch, tokens, channels, order + 1):
        return None
    if gamma.shape != x.shape:
        return None
    if memory_write.shape != (batch, tokens, memory):
        return None
    # THE READ DIRECTION IS PER STAGE: nu^(k) for k = 0..K, so memory_read is [B, T, K+1, M].
    if memory_read.shape != (batch, tokens, order + 1, memory):
        return None
    tensors = (x, raw_alpha, raw_rho, gamma, memory_write, memory_read)
    if not all(tensor.is_cuda for tensor in tensors):
        return None
    if len({tensor.device for tensor in tensors}) != 1:
        return None
    if len({tensor.dtype for tensor in tensors}) != 1:
        return None
    suffix = SUFFIX.get(str(x.dtype))
    stage = ARCH_STAGE[arch]
    if suffix is None:
        return None
    if not memory_write.is_contiguous():
        return None
    # THE RADIUS COEFFICIENT IS REQUIRED: the canonical stage is r R(theta), so a call without it
    # is refused rather than run as the undamped rotation.
    if not (raw_beta is not None and raw_beta.shape == raw_alpha.shape
            and raw_beta.dtype == raw_alpha.dtype and raw_beta.device == x.device):
        return None
    row_stride = _row_stride(x, tokens)
    gamma_row_stride = _row_stride(gamma, tokens)
    if (row_stride is None or row_stride < channels or gamma_row_stride is None
            or gamma_row_stride < channels):
        return None
    alpha_row_stride = dense_tail_row_stride(raw_alpha, tokens)
    beta_row_stride = dense_tail_row_stride(raw_beta, tokens)
    rho_row_stride = dense_tail_row_stride(raw_rho, tokens)
    read_row_stride = dense_tail_row_stride(memory_read, tokens)
    if None in (alpha_row_stride, beta_row_stride, rho_row_stride, read_row_stride):
        return None
    if not status(x.device.index)[0]:
        return None
    if FUNCTION is None:
        FUNCTION = make_function()
    if valid is not None:
        if valid.shape != (batch, tokens) or valid.device != x.device or valid.dtype != torch.bool:
            raise ValueError("mira-dsp mask must be a boolean [batch, tokens] CUDA tensor")
        valid = valid.contiguous()
    if state is not None:
        if (state.shape != (batch, order, channels, memory) or state.device != x.device
                or state.dtype != torch.float32):
            raise ValueError("mira-dsp state must be FP32 [batch, order, channels, memory]")
        if state.requires_grad:
            raise ValueError("mira-dsp cached state is inference-only")
        if not return_state:
            raise ValueError("mira-dsp initial state requires return_state=True")
        state = state.contiguous()
    packed_input = packed_coefficients = None
    if packed is not None:
        packed_input, packed_coefficients = packed
        if (packed_input.shape != (batch, tokens, channels + memory)
                or packed_coefficients.shape != (
                    batch, tokens, channels * (3 * order + 1) + (order + 1) * memory)
                or not packed_input.is_contiguous() or not packed_coefficients.is_contiguous()
                or any(value.device != x.device or value.dtype != x.dtype
                       for value in packed)):
            raise ValueError("mira-dsp packed projections have incompatible layout")
        input_base, coefficient_base = packed_input.data_ptr(), packed_coefficients.data_ptr()
        element = x.element_size()
        if (x.data_ptr() != input_base
                or raw_alpha.data_ptr() != coefficient_base
                or raw_beta.data_ptr() != coefficient_base + channels * order * element
                or raw_rho.data_ptr() != coefficient_base + 2 * channels * order * element
                or memory_read.data_ptr() != coefficient_base + channels * (3 * order + 1) * element
                or row_stride != channels + memory
                or any(stride != packed_coefficients.shape[-1] for stride in
                       (alpha_row_stride, beta_row_stride, rho_row_stride, read_row_stride))):
            raise ValueError("mira-dsp packed views do not match their projection storage")
    keep = bool(not return_state and torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in tensors + (raw_beta,)
        + (() if packed is None else packed)))
    index = x.device.index or 0
    module = D.module_for(CUBIN, index, FNS)
    if module is None:
        return None
    chunk = min(
        _fine_chunk(module, index, order, memory, suffix, batch, tokens, channels, stage)[0]
        if keep else _inference_chunk(
            module, index, order, memory, suffix, batch, tokens, channels),
        tokens)
    if packed is not None:
        x, raw_alpha, raw_beta, raw_rho, memory_write, memory_read = (
            value.detach() for value in (x, raw_alpha, raw_beta, raw_rho, memory_write, memory_read))
    args = (
        x, raw_alpha, raw_beta, raw_rho, gamma, memory_write, memory_read,
        order, memory, chunk, suffix, stage, row_stride,
        gamma_row_stride, alpha_row_stride, beta_row_stride, rho_row_stride,
        read_row_stride, keep,
        packed_input, packed_coefficients, valid, state, bool(return_state))
    if return_state:
        with torch.no_grad():
            return FUNCTION.apply(*args)
    return FUNCTION.apply(*args)


# ------------------------------------------------------------------------------------------------
# mira_v4 / v5: THE INFERENCE SCAN THAT FORMS ITS OWN COEFFICIENTS (mira_iir.cu, fused_scan)
# ------------------------------------------------------------------------------------------------
FUSED_PAIRS = ((2, 16),)
FUSED_RANKS = (16, 32, 48, 64)
FUSED_BATCH = 8
FUSED_CHANNELS = 32
FUSED_TOKENS = 2
FUSED_THREADS = FUSED_BATCH * FUSED_CHANNELS
FUSED_TIMED_RANKS = (48,)
FUSED_TIMING = 8          # values per warp in a _timed kernel's record (mira_iir.cu, fused_scan)
FUSED_TIMING_FIELDS = ("whole", "tile bookkeeping", "barrier A", "form + stage + fetch",
                       "barrier B", "next-token loads + coefficient load + stage map",
                       "recurrence + readout + gate + store")


def _fused_name(order, memory, rank, mode, timed=False):
    return f"mira_fused_scan_k{order}_m{memory}_r{rank}_s{mode}_bf16" + ("_timed" if timed else "")


FNS += tuple(_fused_name(order, memory, rank, mode)
             for order, memory in FUSED_PAIRS for rank in FUSED_RANKS
             for mode in sorted(set(ARCH_STAGE.values())))
FNS += tuple(_fused_name(order, memory, rank, mode, timed=True)
             for order, memory in FUSED_PAIRS for rank in FUSED_TIMED_RANKS
             for mode in sorted(set(ARCH_STAGE.values())))
_PACKED_COEFFICIENTS = {}


def _fused_columns(order):
    return (3 * order + 1 + 7) // 8 * 8


def _fused_shared(order, memory, rank):
    """Dynamic shared bytes of fused_scan: the block's weights, two control tiles, the coefficient
    tile, and two buffers of the float-staged memory_write / memory_read rows (2 tokens x 8 rows)."""
    rows = FUSED_TOKENS * FUSED_BATCH
    columns = _fused_columns(order)
    weights = FUSED_CHANNELS * columns * rank * 2
    control = rows * (rank + 8) * 2
    coefficients = rows * FUSED_CHANNELS * columns * 2
    staged = rows * memory * 4 + rows * (order + 1) * memory * 4
    return weights + 2 * control + coefficients + 2 * staged


def _packed_coefficient_weights(weight, bias, spans, order, channels):
    """[d_e, columns, d_r] bf16 weights and [d_e, columns] fp32 bias, one row of columns per
    channel: alpha_0..alpha_{K-1}, beta_0..beta_{K-1}, rho_0..rho_K, zero padding. A row
    permutation of coefficient_projection (exact), cached per parameter version."""
    import torch

    key = (weight.data_ptr(), weight._version, tuple(weight.shape), weight.dtype, weight.device,
           None if bias is None else (bias.data_ptr(), bias._version), tuple(spans))
    cached = _PACKED_COEFFICIENTS.get(id(weight))
    if cached is not None and cached[0] == key:
        return cached[1], cached[2]
    (alpha_lo, _a), (beta_lo, _b), (rho_lo, _r) = spans
    columns = _fused_columns(order)
    rank = weight.shape[1]
    channel = torch.arange(channels, device=weight.device)
    rows = torch.full((channels, columns), -1, dtype=torch.long, device=weight.device)
    for k in range(order):
        rows[:, k] = alpha_lo + channel * order + k
        rows[:, order + k] = beta_lo + channel * order + k
    for k in range(order + 1):
        rows[:, 2 * order + k] = rho_lo + channel * (order + 1) + k
    used = rows >= 0
    packed = torch.zeros(channels, columns, rank, dtype=weight.dtype, device=weight.device)
    packed[used] = weight.detach()[rows[used]]
    packed_bias = torch.zeros(channels, columns, dtype=torch.float32, device=weight.device)
    if bias is not None:
        packed_bias[used] = bias.detach()[rows[used]].float()
    _PACKED_COEFFICIENTS[id(weight)] = (key, packed.contiguous(), packed_bias.contiguous())
    return _PACKED_COEFFICIENTS[id(weight)][1], _PACKED_COEFFICIENTS[id(weight)][2]


def _fused_plan(module, index, x, control, weight, gamma, memory_write, order, memory, stage,
                timed=False):
    """None, or (rank, row strides) when fused_scan serves this call: inference, one chunk,
    bf16, a compiled (K, d_m, d_r), compute capability 8.0 or later, and the shared block fits."""
    import torch

    if module is None or torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (x, control, weight, gamma, memory_write)):
        return None
    if (order, memory) not in FUSED_PAIRS:
        return None
    rank = int(control.shape[-1])
    if rank not in FUSED_RANKS:
        return None
    name = _fused_name(order, memory, rank, stage, timed)
    if name not in module:
        return None
    if any(tensor.dtype != torch.bfloat16 for tensor in (x, control, weight, gamma, memory_write)):
        return None
    if torch.cuda.get_device_capability(x.device)[0] < 8:
        return None
    batch, tokens, channels = x.shape
    if _inference_chunk(module, index, order, memory, "bf16", batch, tokens, channels) < tokens:
        return None
    if _fused_shared(order, memory, rank) > _hardware(index)["shared_optin_per_block"]:
        return None
    row_stride = _row_stride(x, tokens)
    gamma_row_stride = _row_stride(gamma, tokens)
    if row_stride is None or gamma_row_stride is None:
        return None
    return name, rank, row_stride, gamma_row_stride


def matrix_lattice_iir_control(
    x, control, coefficient_weight, coefficient_bias, spans, gamma, memory_write,
    valid=None, state=None, return_state=False, arch="cauchy_x_hyperbolic",
):
    """The mixer forward from the CONTROL vector rather than the materialised coefficients.

    `control` is SiLU(to_control(z)) [B, T, d_r]; coefficient_projection (weight [rows, d_r],
    bias [rows]) and `spans` -- the (lo, hi) rows of alpha, beta, rho and memory_read -- define
        coefficients = F.linear(control, coefficient_weight, coefficient_bias).
    When fused_scan serves the call (inference, one chunk, bf16, sm_80+) the scan forms alpha,
    beta and rho itself and only memory_read (the (K + 1) d_m read rows) is projected here.
    Otherwise the coefficients are projected exactly as the mixer always has and
    matrix_lattice_iir runs (v3's path). The result is the same function either way; the fused
    path differs only in the order of each coefficient's d_r-term sum."""
    return _matrix_lattice_iir_control(
        x, control, coefficient_weight, coefficient_bias, spans, gamma, memory_write,
        valid=valid, state=state, return_state=return_state, arch=arch, timed=False)


def fused_scan_timing(x, control, coefficient_weight, coefficient_bias, spans, gamma,
                      memory_write, arch="cauchy_x_hyperbolic"):
    """One run of the _timed fused_scan (d_r = 48) on these inputs, or None when fused_scan would
    not serve them. Returns {"fields": FUSED_TIMING_FIELDS, "cycles_per_token": {field: mean over
    warps of a warp's cycles in that phase / tokens}, "warps", "blocks", "tokens",
    "sm_blocks": {SM id: blocks}, "whole_cycles": [min, mean, max] over warps}. The clock marks
    serialise a little around each phase, so the parts are the timed kernel's, not v5's exact
    split; their sum is the timed kernel's whole."""
    import torch

    result = _matrix_lattice_iir_control(
        x, control, coefficient_weight, coefficient_bias, spans, gamma, memory_write,
        arch=arch, timed=True)
    if result is None:
        return None
    _output, record, blocks = result
    batch, tokens, _channels = x.shape
    record = record.view(blocks, FUSED_BATCH, FUSED_TIMING).cpu()
    live = torch.arange(blocks * FUSED_BATCH).view(blocks, FUSED_BATCH)
    groups = blocks // ((batch + FUSED_BATCH - 1) // FUSED_BATCH)
    live = (live // FUSED_BATCH // groups) * FUSED_BATCH + live % FUSED_BATCH < batch
    rows = record[live]
    cycles = rows[:, :FUSED_TIMING - 1].double()
    sms = record[:, 0, FUSED_TIMING - 1]
    per_sm = {}
    for sm in sms.tolist():
        per_sm[int(sm)] = per_sm.get(int(sm), 0) + 1
    return {
        "fields": list(FUSED_TIMING_FIELDS),
        "cycles_per_token": {
            field: float(cycles[:, k].mean()) / tokens
            for k, field in enumerate(FUSED_TIMING_FIELDS)},
        "whole_cycles": [float(cycles[:, 0].min()), float(cycles[:, 0].mean()),
                         float(cycles[:, 0].max())],
        "warps": int(rows.shape[0]),
        "blocks": int(blocks),
        "tokens": int(tokens),
        "batch": int(batch),
        "sm_blocks": per_sm,
    }


def _matrix_lattice_iir_control(
    x, control, coefficient_weight, coefficient_bias, spans, gamma, memory_write,
    valid=None, state=None, return_state=False, arch="cauchy_x_hyperbolic", timed=False,
):
    import torch
    import torch.nn.functional as F

    alpha_span, beta_span, rho_span, read_span = spans
    batch, tokens, channels = x.shape
    order = (alpha_span[1] - alpha_span[0]) // channels
    memory = int(memory_write.shape[-1])
    stage = ARCH_STAGE[arch]
    index = x.device.index or 0
    module = (D.module_for(CUBIN, index, FNS)
              if x.is_cuda and status(x.device.index)[0] else None)
    plan = _fused_plan(module, index, x, control, coefficient_weight, gamma, memory_write,
                       order, memory, stage, timed=timed)
    read_bias = None if coefficient_bias is None else coefficient_bias[read_span[0]:read_span[1]]
    if plan is None:
        if timed:
            return None
        coefficients = F.linear(control, coefficient_weight, coefficient_bias)
        raw_alpha, raw_beta, raw_rho, memory_read = coefficients.split(
            tuple(hi - lo for lo, hi in spans), dim=-1)
        return matrix_lattice_iir(
            x, raw_alpha.view(batch, tokens, channels, order), raw_rho.view(
                batch, tokens, channels, order + 1), gamma, memory_write,
            memory_read.view(batch, tokens, order + 1, memory),
            raw_beta=raw_beta.view(batch, tokens, channels, order), valid=valid, state=state,
            return_state=return_state, arch=arch)
    name, rank, row_stride, gamma_row_stride = plan
    if valid is not None:
        if valid.shape != (batch, tokens) or valid.device != x.device or valid.dtype != torch.bool:
            raise ValueError("mira-dsp mask must be a boolean [batch, tokens] CUDA tensor")
        valid = valid.contiguous()
    if state is not None:
        if (state.shape != (batch, order, channels, memory) or state.device != x.device
                or state.dtype != torch.float32):
            raise ValueError("mira-dsp state must be FP32 [batch, order, channels, memory]")
        if not return_state:
            raise ValueError("mira-dsp initial state requires return_state=True")
        state = state.contiguous()
    control = control.contiguous()
    memory_write = memory_write.contiguous()
    memory_read = F.linear(control, coefficient_weight[read_span[0]:read_span[1]],
                           read_bias).contiguous()
    weights, bias = _packed_coefficient_weights(
        coefficient_weight, coefficient_bias, spans[:3], order, channels)
    output = torch.empty(batch, tokens, channels, device=x.device, dtype=x.dtype)
    final = (torch.empty(batch, order, channels, memory, device=x.device, dtype=torch.float32)
             if return_state else None)
    groups = (channels + FUSED_CHANNELS - 1) // FUSED_CHANNELS
    blocks = ((batch + FUSED_BATCH - 1) // FUSED_BATCH) * groups
    record = (torch.zeros(blocks * FUSED_BATCH * FUSED_TIMING, device=x.device,
                          dtype=torch.int64) if timed else None)
    launch(module, name, index, blocks, FUSED_THREADS,
           [x.data_ptr(), control.data_ptr(), weights.data_ptr(), bias.data_ptr(),
            gamma.data_ptr(), memory_write.data_ptr(), memory_read.data_ptr(), output.data_ptr(),
            0 if state is None else state.data_ptr(), 0 if final is None else final.data_ptr(),
            0 if valid is None else valid.data_ptr(), 0 if record is None else record.data_ptr()],
           [batch, tokens, channels, row_stride, gamma_row_stride, rank,
            (order + 1) * memory, stage],
           shared=_fused_shared(order, memory, rank))
    if timed:
        return output, record, blocks
    USED["forward"] += 1
    return (output, final) if return_state else output
