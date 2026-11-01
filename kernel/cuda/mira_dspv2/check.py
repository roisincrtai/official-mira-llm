"""Value-and-gradient parity for MIRA-Matrix v7 (kernel/cuda/mira_dspv2: v3 tree scan + the
coefficient-forming inference scan), its
dispatch rules, and a line-by-line CPU port of the tree-scan kernel against the serial composition."""

from __future__ import annotations

import importlib
import itertools
import math
import os
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
while ROOT != os.path.dirname(ROOT) and not os.path.isdir(os.path.join(ROOT, "model")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)


# A beta_bias per arch at which the radius is damped but not small: r near 0.64 under the
# cauchy radius.
DAMPED = {f"{radius}_x_{rotation}": value
          for radius, value in (("cauchy", 0.75), ("gaussian", 0.67), ("expexp", -0.81),
                                ("expsoftplus", -0.58), ("hill", 1.33))
          for rotation in ("mobius", "hyperbolic", "identity")}


def _scaled64(got, reference):
    """_scaled in float64: for the CPU emulation, whose float64 differences _scaled's float32 cast
    would round away."""
    scale = float(reference.detach().double().abs().max())
    error = float((got.detach().double() - reference.detach().double()).abs().max())
    return error / scale if scale > 0.0 else (0.0 if error == 0.0 else float("inf"))


def _scaled(got, reference):
    scale = float(reference.detach().float().abs().max())
    error = float((got.detach().float() - reference.detach().float()).abs().max())
    return error / scale if scale > 0.0 else (0.0 if error == 0.0 else float("inf"))


def _case(
    memory, dtype_name, tokens, channels, alpha_bias=0.0, strided=False,
    order=4, beta_bias=0.0, batch=2, arch="cauchy_x_mobius",
):
    import torch

    cuda = importlib.import_module("kernel.cuda.mira_dspv2.mira")
    reference = importlib.import_module("model.mira_mixer")
    dtype = getattr(torch, dtype_name)
    device = torch.device("cuda")
    torch.manual_seed(20260830 + memory + tokens + channels)
    shapes = (
        (batch, tokens, channels),
        (batch, tokens, channels, order),
        (batch, tokens, channels, order + 1),
        (batch, tokens, channels),
        (batch, tokens, memory),
        (batch, tokens, order + 1, memory),
    )
    scales = (0.15, 0.12, 0.15, 0.15, 0.12, 0.12)
    source = [torch.randn(*shape, device=device, dtype=dtype) * scale
              for shape, scale in zip(shapes, scales)]
    source[1].add_(alpha_bias)
    # THE RADIUS COEFFICIENT IS THE SEVENTH TENSOR, so it is appended rather than inserted: the
    # strided case still rewrites entries 0 and 3.
    source.append(
        torch.randn(batch, tokens, channels, order, device=device, dtype=dtype) * 0.12
        + beta_bias)
    def inputs():
        values = [value.detach().clone().requires_grad_(True) for value in source]
        if strided:
            width = 2 * channels + 3
            for index, start in ((0, 0), (3, channels)):
                base = torch.empty(batch, tokens, width, device=device, dtype=dtype)
                view = base[:, :, start:start + channels]
                view.copy_(source[index])
                values[index] = view.detach().requires_grad_(True)
        return values

    kernel_inputs = inputs()
    reference_inputs = inputs()
    got = cuda.matrix_lattice_iir(
        *kernel_inputs[:6], raw_beta=kernel_inputs[6], arch=arch)
    if got is None:
        raise AssertionError(f"CUDA refused memory={memory}, dtype={dtype_name}")
    expected = reference.matrix_lattice_iir(
        *reference_inputs[:6], raw_beta=reference_inputs[6], arch=arch)
    upstream = torch.randn_like(got)
    (got.float() * upstream.float()).sum().backward()
    (expected.float() * upstream.float()).sum().backward()
    tolerance = 3.0e-4 if dtype is torch.float32 else 1.5e-2
    errors = {"output": _scaled(got, expected)}
    names = ("x", "alpha", "rho", "gamma", "memory_write", "memory_read", "beta")
    errors.update({name: _scaled(left.grad, right.grad)
                   for name, left, right in zip(names, kernel_inputs, reference_inputs)})
    failed = {name: error for name, error in errors.items()
              if not math.isfinite(error) or error > tolerance}
    if failed:
        detail = ", ".join(f"{name}={error:.3e}" for name, error in failed.items())
        raise AssertionError(
            f"arch={arch} memory={memory} dtype={dtype_name} "
            f"{detail}; tolerance={tolerance:.3e}")
    return max(errors.values())


def _forward_case(
    memory, dtype_name, tokens, channels, chunk=None, alpha_bias=0.0,
    order=4, beta_bias=0.0, arch="cauchy_x_mobius",
):
    import torch

    cuda = importlib.import_module("kernel.cuda.mira_dspv2.mira")
    reference = importlib.import_module("model.mira_mixer")
    dtype = getattr(torch, dtype_name)
    device = torch.device("cuda")
    torch.manual_seed(20260830 + memory + tokens + channels)
    batch = 2
    shapes = (
        (batch, tokens, channels),
        (batch, tokens, channels, order),
        (batch, tokens, channels, order + 1),
        (batch, tokens, channels),
        (batch, tokens, memory),
        (batch, tokens, order + 1, memory),
    )
    scales = (0.15, 0.12, 0.15, 0.15, 0.12, 0.12)
    source = [torch.randn(*shape, device=device, dtype=dtype) * scale
              for shape, scale in zip(shapes, scales)]
    source[1].add_(alpha_bias)
    beta = (torch.randn(batch, tokens, channels, order, device=device, dtype=dtype) * 0.12
            + beta_bias)
    with torch.no_grad():
        if chunk is None:
            got = cuda.matrix_lattice_iir(
                *source, raw_beta=beta, arch=arch)
        else:
            got = cuda.make_function().apply(
                source[0], source[1], beta, *source[2:], order, memory, chunk,
                cuda.SUFFIX[str(dtype)], cuda.ARCH_STAGE[arch],
                channels, channels, channels * order, channels * order,
                channels * (order + 1), (order + 1) * memory, False)
        expected = reference.matrix_lattice_iir(
            *source, raw_beta=beta, arch=arch)
    if got is None:
        raise AssertionError(f"CUDA refused memory={memory}, dtype={dtype_name}")
    error = _scaled(got, expected)
    tolerance = 3.0e-4 if dtype is torch.float32 else 1.5e-2
    if not math.isfinite(error) or error > tolerance:
        raise AssertionError(
            f"arch={arch} memory={memory} dtype={dtype_name} "
            f"inference={error:.3e}; tolerance={tolerance:.3e}")
    return error



def _packed_case(dtype_name, masked=False, order=4, memory=16,
                 batch=2, tokens=33, channels=131, arch="cauchy_x_mobius", beta_bias=0.0):
    import torch
    import torch.nn.functional as F

    cuda = importlib.import_module("kernel.cuda.mira_dspv2.mira")
    reference = importlib.import_module("model.mira_mixer")
    dtype = getattr(torch, dtype_name)
    torch.manual_seed(20260830)
    source = [
        torch.randn(batch, tokens, channels + memory, device="cuda", dtype=dtype) * 0.12,
        torch.randn(batch, tokens, channels * (3 * order + 1) + (order + 1) * memory,
                    device="cuda", dtype=dtype) * 0.12,
        torch.randn(batch, tokens, channels, device="cuda", dtype=dtype) * 0.12,
    ]
    left = [value.detach().clone().requires_grad_(True) for value in source]
    right = [value.detach().clone().requires_grad_(True) for value in source]
    valid = None
    if masked:
        valid = torch.ones(batch, tokens, device="cuda", dtype=torch.bool)
        valid[0] = False
        valid[1, ::3] = False

    def arguments(values):
        signal, write = values[0].split((channels, memory), dim=-1)
        alpha, beta, rho, read = values[1].split(
            (channels * order, channels * order, channels * (order + 1),
             (order + 1) * memory), dim=-1)
        return (signal, alpha.view(batch, tokens, channels, order),
                rho.view(batch, tokens, channels, order + 1), values[2],
                F.silu(write).contiguous(), read.view(batch, tokens, order + 1, memory),
                beta.view(batch, tokens, channels, order))

    if beta_bias:
        with torch.no_grad():
            source[1][..., channels * order:2 * channels * order].add_(beta_bias)
        left = [value.detach().clone().requires_grad_(True) for value in source]
        right = [value.detach().clone().requires_grad_(True) for value in source]

    args = arguments(left)
    got = cuda.matrix_lattice_iir(*args[:6], raw_beta=args[6],
                                 valid=valid, packed=(left[0], left[1]), arch=arch)
    if got is None:
        raise AssertionError(f"CUDA refused packed K={order}, memory={memory}, dtype={dtype_name}")
    args = arguments(right)
    expected = reference.matrix_lattice_iir(
        *args[:6], raw_beta=args[6], valid=valid, arch=arch)
    upstream = torch.randn_like(got)
    (got.float() * upstream.float()).sum().backward()
    (expected.float() * upstream.float()).sum().backward()
    pairs = [(got, expected)] + [(a.grad, b.grad) for a, b in zip(left, right)]
    for a, b in pairs:
        tolerance = 3e-4 if dtype == torch.float32 else 1.5e-2
        torch.testing.assert_close(a, b, rtol=tolerance,
                                   atol=3e-6 if dtype == torch.float32 else 3e-4,
                                   msg=lambda message: (
                                       f"packed arch={arch} K={order} memory={memory} "
                                       f"dtype={dtype_name} masked={masked}: {message}"))
        error = _scaled(a, b)
        if not math.isfinite(error) or error > tolerance:
            raise AssertionError(
                f"packed K={order} memory={memory} dtype={dtype_name} "
                f"relative error={error:.3e}; tolerance={tolerance:.3e}")
    return max(_scaled(a, b) for a, b in pairs)


def _state_case(order, memory, dtype_name, tokens=35, channels=3, partitions=None,
                arch="cauchy_x_mobius", beta_bias=0.0):
    import torch

    cuda = importlib.import_module("kernel.cuda.mira_dspv2.mira")
    reference = importlib.import_module("model.mira_mixer")
    dtype = getattr(torch, dtype_name)
    batch = 2
    torch.manual_seed(20260830)
    shapes = ((batch, tokens, channels), (batch, tokens, channels, order),
              (batch, tokens, channels, order + 1), (batch, tokens, channels),
              (batch, tokens, memory), (batch, tokens, order + 1, memory),
              (batch, tokens, channels, order))
    values = [torch.randn(*shape, device="cuda", dtype=dtype) * 0.12 for shape in shapes]
    values[6].add_(beta_bias)
    initial = torch.randn(batch, order, channels, memory, device="cuda", dtype=torch.float32) * 0.1
    valid = torch.ones(batch, tokens, device="cuda", dtype=torch.bool)
    valid[0] = False
    valid[1, ::4] = False
    with torch.no_grad():
        expected, expected_state = reference.matrix_lattice_iir(
            *values[:6], raw_beta=values[6], valid=valid, state=initial, return_state=True,
            arch=arch)
        got, final = cuda.matrix_lattice_iir(
            *values[:6], raw_beta=values[6], valid=valid, state=initial, return_state=True,
            arch=arch)
        state = initial
        parts = []
        if partitions is None:
            partitions = (1, 15, 1, 17, 1)
        first = 0
        for length in partitions:
            last = first + length
            args = [value[:, first:last].contiguous() for value in values]
            output, state = cuda.matrix_lattice_iir(
                *args[:6], raw_beta=args[6], valid=valid[:, first:last],
                state=state, return_state=True, arch=arch)
            parts.append(output)
            first = last
        if first != tokens:
            raise AssertionError(f"streaming partitions cover {first} of {tokens} tokens")
        streamed = torch.cat(parts, dim=1)
        pairs = ((got, expected), (streamed, expected), (final, expected_state),
                 (state, expected_state), (state[0], initial[0]))
        for a, b in pairs:
            torch.testing.assert_close(a, b, rtol=3e-4 if dtype == torch.float32 else 1.5e-2,
                                       atol=3e-6 if dtype == torch.float32 else 3e-4)
    return max(_scaled(a, b) for a, b in pairs)


def _boundary_case(chunks, reverse, inference=False, poison_final=False, workers=None,
                   order=2, memory=16, batch=1, channels=1):
    """The tree scan on the GPU against the serial composition (float32, torch), at the worker
    count _tree_plan picks or a forced one."""
    import torch
    from unittest.mock import patch

    cuda = importlib.import_module("kernel.cuda.mira_dspv2.mira")
    torch.manual_seed(20260830)
    phi = torch.randn(batch, chunks, channels, order, order, device="cuda") * 0.03
    phi += 0.8 * torch.eye(order, device="cuda")
    carry = torch.randn(batch, chunks, channels, memory, order, device="cuda") * 0.1
    if poison_final:
        phi[:, -1] = float("nan")
        carry[:, -1] = float("nan")
    initial = None if reverse else torch.randn(batch, order, channels, memory, device="cuda")
    state = (torch.zeros(batch, channels, memory, order, device="cuda") if initial is None
             else initial.permute(0, 2, 3, 1).contiguous())
    expected = torch.empty_like(carry)
    for index in (range(chunks - 1, -1, -1) if reverse else range(chunks)):
        expected[:, index] = state
        if inference and index == chunks - 1:
            break
        matrix = phi[:, index].transpose(-1, -2) if reverse else phi[:, index]
        state = carry[:, index] + torch.einsum("bcij,bcmj->bcmi", matrix, state)
    index = torch.cuda.current_device()
    module = cuda.D.module_for(cuda.CUBIN, index, cuda.FNS)
    actual = carry.clone()
    original = cuda._tree_plan

    def forced(*args):
        plan = dict(original(*args))
        if workers is not None:
            plan["workers"] = workers
            plan["blocks"] = -(-plan["units"] * workers // cuda.TREE_THREADS)
        return plan

    with patch.object(cuda, "_tree_plan", side_effect=forced):
        cuda._compose_boundaries(module, index, phi, actual, batch, chunks, channels, order,
                                 memory, initial, reverse, inference)
    torch.cuda.synchronize()
    if inference and not torch.isfinite(actual).all():
        raise AssertionError("the tree scan read the last chunk's summary in inference")
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-6)
    return _scaled(actual, expected)


def _mask_dispatch_case(dtype_name):
    import torch

    cuda = importlib.import_module("kernel.cuda.mira_dspv2.mira")
    dtype = getattr(torch, dtype_name)
    torch.manual_seed(20260919)
    batch, tokens, channels, order, memory = 2, 257, 193, 2, 16
    shapes = ((batch, tokens, channels), (batch, tokens, channels, order),
              (batch, tokens, channels, order + 1), (batch, tokens, channels),
              (batch, tokens, memory), (batch, tokens, order + 1, memory),
              (batch, tokens, channels, order))
    values = [torch.randn(*shape, device="cuda", dtype=dtype) * 0.12 for shape in shapes]
    initial = torch.randn(batch, order, channels, memory, device="cuda") * 0.1
    valid = torch.ones(batch, tokens, device="cuda", dtype=torch.bool)
    with torch.no_grad():
        unmasked, unmasked_state = cuda.matrix_lattice_iir(
            *values[:6], raw_beta=values[6], state=initial, return_state=True)
        masked, masked_state = cuda.matrix_lattice_iir(
            *values[:6], raw_beta=values[6], valid=valid,
            state=initial, return_state=True)
    if not torch.equal(unmasked, masked) or not torch.equal(unmasked_state, masked_state):
        torch.testing.assert_close(unmasked, masked, rtol=0.0, atol=0.0)
        torch.testing.assert_close(unmasked_state, masked_state, rtol=0.0, atol=0.0)
    return max(_scaled(unmasked, masked), _scaled(unmasked_state, masked_state))


def _read_only_gradient_case():
    import torch

    cuda = importlib.import_module("kernel.cuda.mira_dspv2.mira")
    reference = importlib.import_module("model.mira_mixer")
    torch.manual_seed(20260830)
    batch, tokens, channels, order, memory = 1, 17, 3, 4, 16
    shapes = ((batch, tokens, channels), (batch, tokens, channels, order),
              (batch, tokens, channels, order + 1), (batch, tokens, channels),
              (batch, tokens, memory), (batch, tokens, order + 1, memory),
              (batch, tokens, channels, order))
    values = [torch.randn(*shape, device="cuda") * 0.12 for shape in shapes]
    left = values[5].detach().clone().requires_grad_(True)
    right = values[5].detach().clone().requires_grad_(True)
    cuda.matrix_lattice_iir(*values[:5], left, raw_beta=values[6]).sum().backward()
    reference.matrix_lattice_iir(*values[:5], right, raw_beta=values[6]).sum().backward()
    torch.testing.assert_close(left.grad, right.grad, rtol=3e-4, atol=3e-6)
    return _scaled(left.grad, right.grad)


def _projection_case(dtype_name):
    """Packed eval projection parity, cache invalidation, and strided-gamma CUDA parity."""
    import torch
    import torch.nn.functional as F

    cuda = importlib.import_module("kernel.cuda.mira_dspv2.mira")
    reference = importlib.import_module("model.mira_mixer")
    dtype = getattr(torch, dtype_name)
    torch.manual_seed(20260919)
    batch, tokens, width, channels, order, memory, bottleneck = 2, 33, 32, 193, 2, 16, 8
    mixer = reference.MiraMixer(
        width, lattice_order=order, d_e=channels,
        coefficient_bottleneck=bottleneck, memory_dimension=memory,
        dropout=0.0).to(device="cuda", dtype=dtype).eval()
    normalized = torch.randn(batch, tokens, width, device="cuda", dtype=dtype)
    state_keys = tuple(mixer.state_dict())

    def original_projections():
        control = F.silu(mixer.to_control(normalized))
        return (mixer.project_input(normalized),
                mixer.project_coefficients(control), mixer.to_gate(normalized))

    with torch.no_grad():
        expected = original_projections()
        projected_input, projected_coefficients, gamma = (
            mixer.project_recurrence_inputs(normalized, tokens))
        actual = (projected_input, projected_coefficients, gamma)
        errors = [_scaled(left, right) for left, right in zip(actual, expected)]
        packed = getattr(mixer, "_packed_input_projection", None)
        if packed is None:
            raise AssertionError("eval/no-grad projections did not create their packed cache")
        reused = mixer.project_recurrence_inputs(normalized, tokens)
        if mixer._packed_input_projection is not packed:
            raise AssertionError("packed projection cache missed without a parameter change")
        errors.extend(_scaled(left, right) for left, right in zip(reused, expected))
        if gamma.is_contiguous():
            raise AssertionError("packed gamma is dense, so the CUDA row-stride path is untested")

        signal, raw_write = projected_input.split((channels, memory), dim=-1)
        alpha, beta, rho, read = projected_coefficients.split(
            (channels * order, channels * order, channels * (order + 1),
             (order + 1) * memory), dim=-1)
        alpha = alpha.view(batch, tokens, channels, order)
        beta = beta.view(batch, tokens, channels, order)
        rho = rho.view(batch, tokens, channels, order + 1)
        read = read.view(batch, tokens, order + 1, memory)
        write = F.silu(raw_write).contiguous()
        strided = cuda.matrix_lattice_iir(
            signal, alpha, rho, gamma, write, read, raw_beta=beta)
        dense = cuda.matrix_lattice_iir(
            signal, alpha, rho, gamma.contiguous(), write, read, raw_beta=beta)
        if strided is None or dense is None:
            raise AssertionError(
                f"CUDA refused packed projection K={order}, memory={memory}, dtype={dtype_name}")
        errors.append(_scaled(strided, dense))

        old_cache = mixer._packed_input_projection
        mixer.to_gate.bias.add_(0.125)
        updated = mixer.project_recurrence_inputs(normalized, tokens)
        if mixer._packed_input_projection is old_cache:
            raise AssertionError("packed projection cache survived an in-place parameter update")
        errors.extend(_scaled(left, right)
                      for left, right in zip(updated, original_projections()))
        if tuple(mixer.state_dict()) != state_keys:
            raise AssertionError("packed projection cache changed the checkpoint schema")
        if any("packed_input_projection" in key for key in mixer.state_dict()):
            raise AssertionError("packed projection cache entered the checkpoint")

    tolerance = 3.0e-4 if dtype is torch.float32 else 1.5e-2
    error = max(errors)
    if not math.isfinite(error) or error > tolerance:
        raise AssertionError(
            f"packed eval projection dtype={dtype_name} error={error:.3e}; "
            f"tolerance={tolerance:.3e}")
    return error


def check_map():
    from unittest.mock import patch
    import torch

    cuda_map = importlib.import_module("kernel.cuda.mira_dspv2.mira")
    names = set(cuda_map.FNS)
    for (memory, order), (lanes, group) in cuda_map.BACKWARD_MAP.items():
        for suffix in ("f32", "bf16", "f16"):
            name = f"mira_matrix_scan_b3_k{order}_m{memory}_l{lanes}_g{group}_{suffix}"
            if name not in names:
                raise AssertionError(f"BACKWARD_MAP names {name}, which is not instantiated")
        # THE SHARED BLOCK AT THE CHUNK v2 CHOOSES IS CHECKED PER GPU, in check_dispatch.
        scan_lanes, scan_group = cuda_map._scan_plan(
            None, 0, order, memory, "bf16", 1, 1, 1)
        for phase, suffix in (("p1", "_bf16"), ("p3", "_bf16"), ("b1", "_bf16")):
            name = (f"mira_matrix_scan_{phase}_k{order}_m{memory}"
                    f"_l{scan_lanes}_g{scan_group}{suffix}")
            if name not in names:
                raise AssertionError(f"_scan_plan names {name}, which is not instantiated")
        p1_lanes, p1_group = cuda_map._phase_one_variants(memory, order)[-1]
        for suffix in ("f32", "bf16", "f16"):
            name = f"mira_matrix_scan_p1_k{order}_m{memory}_l{p1_lanes}_g{p1_group}_{suffix}"
            if name not in names:
                raise AssertionError(
                    f"_phase_one_variants names {name}, which is not instantiated")
        inference_lanes, inference_group = cuda_map._inference_scan_plan(order, memory)
        if order == 2 and memory == 16 and (inference_lanes, inference_group) != (2, 192):
            raise AssertionError(
                "K=2 d_m=16 inference must use 2 lanes x 192 channels")
        replay_lanes, replay_group = cuda_map._inference_replay_plan(order, memory)
        if order == 2 and memory == 16 and (replay_lanes, replay_group) != (1, 128):
            raise AssertionError(
                "K=2 d_m=16 inference replay must use 1 lane x 128 channels")
        for phase, plan_lanes, plan_group in (
                ("p1", inference_lanes, inference_group),
                ("p3", replay_lanes, replay_group)):
            for suffix in ("f32", "bf16", "f16"):
                name = cuda_map._forward_scan_name(
                    phase, order, memory, plan_lanes, plan_group, suffix, False)
                if name not in names:
                    raise AssertionError(
                        f"_inference_scan_plan names {name}, which is not instantiated")
        # THE COMPOSITION IS ONE TREE-SCAN KERNEL PER (K, d_m), launched once at any chunk count.
        name = cuda_map._tree_name(order, memory)
        if name not in names:
            raise AssertionError(f"_tree_name names {name}, which is not instantiated")
        if memory % cuda_map._tree_columns(order, memory):
            raise AssertionError(f"tree columns do not tile d_m={memory}")
    if any(name.startswith(("mira_matrix_scan_p2", "mira_matrix_scan_b2", "mira_matrix_affine"))
           for name in names):
        raise AssertionError("mira_dspv2 must not load the serial composition kernels")

    with patch.object(cuda_map, "_kernel_resources",
                      side_effect=lambda module, index, name, *args, **kwargs: name), \
            patch.object(cuda_map, "launch") as launched:
        for order, memory in cuda_map._built_pairs():
            for inference in (False, True):
                for chunks in (1, 17, 127, 128, 129, 2051):
                    phi = torch.empty(chunks, order, order, device="cpu")
                    carry = torch.empty(chunks, memory, order, device="cpu")
                    launched.reset_mock()
                    cuda_map._compose_boundaries(
                        None, 0, phi, carry, 1, chunks, 1, order, memory, inference=inference)
                    actual = [call.args[1] for call in launched.call_args_list]
                    details = cuda_map._composition_details(
                        None, 0, 1, chunks, 1, order, memory, inference=inference)
                    if actual != [details["resources"]] or details["method"] != "tree":
                        raise AssertionError(
                            f"composition report names {details['resources']}, launched {actual}")
                    scalars = launched.call_args_list[0].args[6]
                    if scalars[-1] != int(inference) or scalars[3] != details["workers"]:
                        raise AssertionError(f"tree launch scalars {scalars} disagree with the plan")

    check_dispatch(cuda_map, quick=bool(os.environ.get("MIRA_V4_QUICK_DISPATCH")))

    with patch.object(cuda_map.D, "prefer_shared") as preferred, \
            patch.object(cuda_map.D, "launch_dims", return_value=(True, "")) as launched:
        for name in cuda_map.FNS:
            module = {name: name, "__ctx__": None}
            preferred.reset_mock()
            launched.reset_mock()
            cuda_map.launch(module, name, 0, 3, 64, [1, 2], [3, 4], shared=1024)
            if name.startswith(("mira_matrix_scan_p1i_", "mira_matrix_scan_p3i_")):
                preferred.assert_called_once_with(name, None, 1024, percent=-1)
            else:
                preferred.assert_called_once_with(name, None, 1024)
        # THE CEILING NEVER FALLS: a smaller launch after a larger one keeps the larger opt-in, so the
        # larger size, cached by D.prefer_shared, still fits when it comes back.
        name = cuda_map._scan_name("b3", 2, 64, 32, 16, "f32")
        module = {name: name, "__ctx__": None}
        asked = []
        preferred.side_effect = lambda fn, ctx, size, **kwargs: asked.append(size)
        for size in (99840, 54784, 99840, 32256):
            cuda_map.launch(module, name, 0, 8, 512, [1], [2], shared=size)
        preferred.side_effect = None
        if asked != [99840, 99840, 99840, 99840]:
            raise AssertionError(f"the dynamic-shared opt-in fell: {asked}")
            launched.assert_called_once_with(
                name, None, 0, 3, 64, [1, 2], [3, 4], shared=1024)


# mira_v2: THE DISPATCH RULES ON A TABLE OF GPUS, WITHOUT A GPU.
#
# (SMs, shared memory opt-in per block, shared memory per SM, reserved per block), in bytes, as the
# driver reports them on each card. They are inputs to the rules here, not measurements: the point
# is that every GPU of every generation the cubin is built for gets a valid mapping.
DISPATCH_DEVICES = {
    "T4 (sm_75)": (40, 65536, 65536, 0),
    "A10 (sm_86)": (72, 101376, 102400, 1024),
    "A100 (sm_80)": (108, 166912, 167936, 1024),
    "RTX 6000 Ada (sm_89)": (142, 101376, 102400, 1024),
    "H100 SXM (sm_90)": (132, 232448, 233472, 1024),
    "H200 (sm_90)": (132, 232448, 233472, 1024),
    "B200 (sm_100)": (148, 232448, 233472, 1024),
    "RTX 5090 (sm_120)": (170, 101376, 102400, 1024),
    "RTX PRO 6000 Blackwell (sm_120)": (188, 101376, 102400, 1024),
    "DGX Spark GB10 (sm_121)": (48, 101376, 102400, 1024),
}


def check_dispatch(cuda_map, quick=False):
    """Every rule of mira_v2 on every GPU above, every compiled (K, d_m), both dtypes, and a sweep
    of batch, tokens, channels and occupancy: the mappings are valid and fit the GPU."""
    from unittest.mock import patch

    stage = cuda_map.ARCH_STAGE["cauchy_x_hyperbolic"]
    names = set(cuda_map.FNS)
    batches = (1, 2, 19, 256) if quick else (1, 2, 8, 19, 64, 256)
    token_counts = ((1, 17, 300, 2048, 32768) if quick else
                    (1, 7, 17, 64, 300, 512, 2048, 8192, 32768, 131072))
    channel_counts = (35, 1536) if quick else (35, 1024, 1536)
    occupancies = (None, 1, 3) if quick else (None, 1, 2, 3, 6)
    cases = splits = 0
    fine_by_device = {}
    for device, (sms, optin, per_sm, reserved) in DISPATCH_DEVICES.items():
        attributes = {cuda_map.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: per_sm,
                      cuda_map.CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK: reserved}
        for occupancy in occupancies:
            cuda_map._HARDWARE.clear()
            cuda_map._OCCUPANCY.clear()
            blocks = (cuda_map._blocks_per_sm if occupancy is None else
                      (lambda module, index, name, threads, shared, value=occupancy: value))
            with patch.object(cuda_map.D, "sm_count", return_value=sms), \
                    patch.object(cuda_map.D, "max_optin_shared", return_value=optin), \
                    patch.object(cuda_map.D, "device_attr",
                                 side_effect=lambda index, attr, default=0: attributes.get(
                                     attr, default)), \
                    patch.object(cuda_map, "_blocks_per_sm", side_effect=blocks):
                for order, memory in cuda_map._built_pairs():
                    for suffix in ("f32", "bf16"):
                        lanes, group, f_smem, sigma = cuda_map._b3_budget(
                            None, 0, order, memory, suffix, stage)
                        if sigma > optin or f_smem < 8:
                            raise AssertionError(f"{device}: b3 budget {sigma} B, f_smem {f_smem}")
                        if f"mira_matrix_scan_b3_k{order}_m{memory}_l{lanes}_g{group}_{suffix}" \
                                not in names:
                            raise AssertionError(f"{device}: b3 group {group} is not compiled")
                        if occupancy is None and suffix == "bf16":
                            fine_by_device.setdefault(device, {})[(order, memory)] = (
                                cuda_map._pow2floor(f_smem))
                        for batch, tokens, channels in itertools.product(
                                batches, token_counts, channel_counts):
                            fine, b3_lanes, b3_group, shared = cuda_map._fine_chunk(
                                None, 0, order, memory, suffix, batch, tokens, channels, stage)
                            if fine & (fine - 1) or not 8 <= fine <= max(8, f_smem):
                                raise AssertionError(f"{device}: backward chunk {fine}")
                            if shared > sigma:
                                raise AssertionError(
                                    f"{device}: b3 shared {shared} B over the budget {sigma} B")
                            used = min(fine, tokens)
                            plan = cuda_map._backward_plan(
                                None, 0, order, memory, suffix, batch, tokens, channels, used, stage)
                            if plan[:2] != (b3_lanes, b3_group):
                                raise AssertionError(f"{device}: backward plan {plan} differs")
                            for keep in (True, False):
                                chunk = (cuda_map._forward_chunk(
                                    None, 0, order, memory, suffix, batch, tokens, channels,
                                    True, used) if keep else cuda_map._inference_chunk(
                                    None, 0, order, memory, suffix, batch, tokens, channels))
                                align = used if keep else cuda_map.ALPHA_TILE
                                chunks = -(-tokens // chunk)
                                if not (chunk == tokens or chunk % align == 0):
                                    raise AssertionError(
                                        f"{device}: chunk {chunk} is not a multiple of {align}")
                                if not (chunks - 1) * chunk < tokens <= chunks * chunk:
                                    raise AssertionError(f"{device}: {chunks} x {chunk} != {tokens}")
                                if chunks > 1 and chunks > math.isqrt(tokens):
                                    raise AssertionError(f"{device}: {chunks} chunks > sqrt({tokens})")
                                if batch * chunks * channels >= 2 ** 31:
                                    raise AssertionError(f"{device}: grid too large")
                                splits += chunks > 1
                                cases += 1
                                for n in {chunks, -(-tokens // used)}:
                                    plan = cuda_map._tree_plan(None, 0, batch, n, channels,
                                                               order, memory)
                                    w = plan["workers"]
                                    if w & (w - 1) or not 1 <= w <= min(
                                            cuda_map.TREE_MAX_WORKERS, cuda_map._pow2ceil(n)):
                                        raise AssertionError(f"{device}: tree workers {w} for {n}")
                                    if plan["blocks"] * cuda_map.TREE_THREADS < plan["units"] * w \
                                            or plan["blocks"] >= 2 ** 31:
                                        raise AssertionError(f"{device}: tree grid {plan}")
    cuda_map._HARDWARE.clear()
    cuda_map._OCCUPANCY.clear()
    print(f"PASS: mira_dspv2 dispatch on {len(DISPATCH_DEVICES)} GPUs, {cases:,} (GPU, occupancy, K, "
          f"d_m, dtype, batch, tokens, channels, training/inference) cases, {splits:,} split")
    for device, row in fine_by_device.items():
        print(f"      {device:<34s} longest backward chunk, bf16: "
              + ", ".join(f"K={o} d_m={m}: {f}" for (o, m), f in sorted(row.items())))
    return cases


def _fine_chunks(cuda, order, memory, suffix):
    """The backward chunks this GPU's b3 budget allows for (K, d_m): powers of two from 8 tokens."""
    import torch

    index = torch.cuda.current_device()
    module = cuda.D.module_for(cuda.CUBIN, index, cuda.FNS)
    stage = cuda.ARCH_STAGE["cauchy_x_mobius"]
    _lanes, _group, f_smem, _sigma = cuda._b3_budget(module, index, order, memory, suffix, stage)
    return [fine for fine in (8, 16, 32, 64, 128, 256) if fine <= f_smem]


def _fine_case(order, memory, dtype_name, fine):
    """Values and every gradient against the reference with the backward chunk forced to `fine`
    tokens, over three fine chunks and a ragged fourth, so b3 runs at the chunk lengths the larger
    shared memory of this GPU makes possible."""
    from unittest.mock import patch

    cuda = importlib.import_module("kernel.cuda.mira_dspv2.mira")
    original = cuda._fine_chunk

    def forced(module, index, order_, memory_, suffix, batch, tokens, channels, stage_mode):
        _f, lanes, group, _shared = original(
            module, index, order_, memory_, suffix, batch, tokens, channels, stage_mode)
        element_size = 4 if suffix == "f32" else 2
        return fine, lanes, group, cuda._backward_shared(
            order_, memory_, group, fine, stage_mode, element_size)

    with patch.object(cuda, "_fine_chunk", side_effect=forced):
        return _case(memory, dtype_name, 3 * fine + 5, 35, order=order, beta_bias=0.75)


# mira_dspv2: A LINE-BY-LINE CPU PORT OF tree_scan (mira_iir.cu), IN THE KERNEL'S LAYOUTS.
def _tree_emulate(phi, carry, initial, batch, chunks, channels, order, memory, workers,
                  reverse=False, inference=False):
    """float64 numpy. phi [planes K K], carry [planes M K] (planes = (b chunks + chunk) C + c),
    initial [B K C M] or None. Returns the carry buffer after the in-place tree scan: every unit
    (sequence, channel, slice of V columns), every worker's fold of its run of chunks, the
    Kogge-Stone steps inside warp segments of width min(workers, 32), the earlier warps' totals,
    and the replay -- the kernel's order of compositions, vectorised over units only."""
    import numpy as np

    K, M = order, memory
    V = max(1, min(M, 8 // K))
    slices = M // V
    units = batch * channels * slices
    unit = np.arange(units)
    column = (unit % slices) * V
    channel = (unit // slices) % channels
    sequence = unit // (slices * channels)
    phi = np.asarray(phi, dtype=np.float64).reshape(-1, K, K)
    source = np.asarray(carry, dtype=np.float64).reshape(-1, M, K)
    out = source.copy()
    per = -(-chunks // workers)
    readable = chunks - 1 if inference else chunks
    width = min(workers, 32)
    columns = column[:, None] + np.arange(V)[None, :]

    def plane_of(logical):
        physical = chunks - 1 - logical if reverse else logical
        return (sequence * chunks + physical) * channels + channel

    def load(logical):
        planes = plane_of(logical)
        a = phi[planes]
        if reverse:
            a = np.swapaxes(a, 1, 2)
        c = np.swapaxes(source[planes[:, None], columns, :], 1, 2)      # [U, K, V]
        return a, c

    def identity():
        return np.broadcast_to(np.eye(K), (units, K, K)).copy(), np.zeros((units, K, V))

    def after(first, second):                                            # second o first
        return second[0] @ first[0], second[0] @ first[1] + second[1]

    x0 = np.zeros((units, K, V))
    if initial is not None and not reverse:
        init = np.asarray(initial, dtype=np.float64).reshape(batch, K, channels, M)
        x0 = np.swapaxes(init[sequence[:, None], :, channel[:, None], columns], 1, 2)
    starts = [x0] * workers
    if workers > 1:
        maps = []
        for worker in range(workers):
            m = identity()
            for logical in range(worker * per, min(chunks, (worker + 1) * per, readable)):
                m = after(m, load(logical))
            maps.append(m)
        inclusive = list(maps)
        delta = 1
        while delta < width:
            previous = list(inclusive)
            for worker in range(workers):
                if worker % width >= delta:
                    inclusive[worker] = after(previous[worker - delta], previous[worker])
            delta <<= 1
        for worker in range(workers):
            x = x0
            if workers > 32:
                warp_map = identity()
                for earlier in range(worker // 32):
                    warp_map = after(warp_map, inclusive[earlier * 32 + 31])
                x = warp_map[0] @ x + warp_map[1]
            if worker % width:
                exclusive = inclusive[worker - 1]
                x = exclusive[0] @ x + exclusive[1]
            starts[worker] = x
    for worker in range(workers):
        x = starts[worker]
        for logical in range(worker * per, min(chunks, (worker + 1) * per)):
            planes = plane_of(logical)
            if logical >= readable:
                out[planes[:, None], columns, :] = np.swapaxes(x, 1, 2)
                break
            a, c = load(logical)
            out[planes[:, None], columns, :] = np.swapaxes(x, 1, 2)
            x = a @ x + c
    return out


def _serial_compose(phi, carry, initial, batch, chunks, channels, order, memory, reverse,
                    inference):
    """v2's serial composition (scan_phase_two / backward_phase_two) in the same layouts."""
    import numpy as np

    K, M = order, memory
    phi = np.asarray(phi, dtype=np.float64).reshape(batch, chunks, channels, K, K)
    carry = np.asarray(carry, dtype=np.float64).reshape(batch, chunks, channels, M, K)
    out = carry.copy()
    state = np.zeros((batch, channels, M, K))
    if initial is not None and not reverse:
        state = np.transpose(np.asarray(initial, dtype=np.float64).reshape(batch, K, channels, M),
                             (0, 2, 3, 1)).copy()
    for index in (range(chunks - 1, -1, -1) if reverse else range(chunks)):
        out[:, index] = state
        if inference and index == chunks - 1:
            break
        matrix = np.swapaxes(phi[:, index], -1, -2) if reverse else phi[:, index]
        state = carry[:, index] + np.einsum("bcij,bcmj->bcmi", matrix, state)
    return out.reshape(-1, M, K)


def check_tree_cpu(quick=False):
    """The tree-scan port against the serial composition, float64: every built (K, d_m), chunk
    counts across the warp and block edges, every worker count up to 256, forward with and without
    an initial state, backward, and inference with the last summary poisoned by NaN."""
    import numpy as np

    cuda_map = importlib.import_module("kernel.cuda.mira_dspv2.mira")
    rng = np.random.default_rng(20261011)
    counts = ((1, 2, 3, 5, 31, 33, 64, 129, 300) if quick else
              (1, 2, 3, 5, 7, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 257, 300, 1000, 2051))
    worst = 0.0
    cases = 0
    for order, memory in cuda_map._built_pairs():
        batch, channels = 2, 3
        for chunks in counts:
            planes = batch * chunks * channels
            phi = rng.standard_normal((planes, order, order)) * 0.03 + 0.8 * np.eye(order)
            carry = rng.standard_normal((planes, memory, order)) * 0.1
            initial = rng.standard_normal((batch, order, channels, memory))
            workers = 1
            while workers <= min(256, cuda_map._pow2ceil(chunks)):
                for reverse, inference, start in ((False, False, None), (False, False, initial),
                                                  (True, False, None), (False, True, initial)):
                    source = carry.copy()
                    table = phi.copy()
                    if inference:
                        last = (np.arange(batch)[:, None] * chunks + chunks - 1) * channels \
                            + np.arange(channels)[None, :]
                        source.reshape(-1, memory, order)[last.ravel()] = np.nan
                        table.reshape(-1, order, order)[last.ravel()] = np.nan
                    got = _tree_emulate(table, source, start, batch, chunks, channels, order,
                                        memory, workers, reverse, inference)
                    want = _serial_compose(table, source, start, batch, chunks, channels, order,
                                           memory, reverse, inference)
                    if not np.isfinite(got).all():
                        raise AssertionError(
                            f"tree K={order} d_m={memory} n={chunks} workers={workers}: non-finite"
                            f" (inference={inference}: the last summary was read)")
                    error = float(np.abs(got - want).max() / max(1.0, np.abs(want).max()))
                    if error > 1e-12:
                        raise AssertionError(
                            f"tree K={order} d_m={memory} n={chunks} workers={workers} "
                            f"reverse={reverse} inference={inference}: error {error:.3e}")
                    worst = max(worst, error)
                    cases += 1
                workers *= 2
    print(f"PASS: mira_dspv2 tree-scan CPU port vs serial composition (fp64), {cases} cases "
          f"(every (K, d_m), {len(counts)} chunk counts, workers 1..256, forward/initial/backward/"
          f"inference with NaN-poisoned last summary): worst {worst:.2e}")
    return worst


def _tree_boundaries(phis, carries, state, reverse, inference):
    """The emulation's composition through the tree port, at the most workers the chunk count
    allows: the boundary entering each chunk, as [B, C, K, M] tensors."""
    import numpy as np
    import torch

    chunks = len(phis)
    batch, channels, order, memory = carries[0].shape
    dtype = carries[0].dtype
    if chunks == 1:                       # one chunk: its boundary is the opening state
        return [torch.zeros_like(carries[0]) if state is None or reverse else state]
    phi = torch.stack(phis, 1).detach().double().numpy()                  # [B, n, C, K, K]
    carry = torch.stack(carries, 1).transpose(-1, -2).contiguous().detach().double().numpy()
    initial = (None if state is None else
               state.permute(0, 2, 1, 3).contiguous().detach().double().numpy())
    workers = min(256, 1 << max(0, chunks - 1).bit_length())
    out = _tree_emulate(phi, carry, initial, batch, chunks, channels, order, memory, workers,
                        reverse, inference)
    out = torch.from_numpy(out.reshape(batch, chunks, channels, memory, order)).transpose(-1, -2)
    out = out.to(dtype)
    return [out[:, index] for index in range(chunks)]


# mira_v2: THE CHUNKED COMPUTATION AT THE CHUNKS v2 CHOOSES, EMULATED ON THE CPU
# (mira_dspv2: with the composition done by the tree-scan port).
#
# The GPU cases above run the kernels. This runs WITHOUT a GPU: it repeats, in PyTorch, what the
# kernels do with the token axis -- p1 (each coarse chunk from a zero carry: its transition product
# Phi, its carry, and the transition product of every fine chunk inside it), the tree scan (the
# coarse chunks joined by _tree_emulate, the kernel's own order of compositions), p3 (each coarse chunk replayed from its
# boundary, writing the state at every fine edge), and the backward (b1: every fine chunk's adjoint
# from a zero future; the reverse tree scan: those joined last to first through the transposed
# fine Phi; b3: every fine
# chunk's vector-Jacobian product from its checkpoint and its adjoint) -- at the coarse and fine
# chunks _forward_chunk and _fine_chunk pick on the GPUs of DISPATCH_DEVICES, and compares values and
# every gradient with the reference. The per-token map is the reference's own (stage_system).

def _token_maps(reference, alpha, beta, valid, arch):
    import torch

    A, b, Cm, d = reference.stage_system(alpha, beta, arch)
    if valid is not None:
        keep = valid[:, :, None].to(A.dtype)
        eye = torch.eye(A.shape[-1], dtype=A.dtype)
        A = keep[..., None, None] * A + (1 - keep[..., None, None]) * eye
        b = keep[..., None] * b
    return A, b, Cm, d


def _emulated_chunk(maps, inputs, valid, first, last, state):
    """p3 over tokens [first, last): outputs [B, last-first, C] and the state after the last token."""
    import torch
    import torch.nn.functional as F

    A, b, Cm, d, u = maps
    outputs = []
    for token in range(first, last):
        levels = Cm[:, token] @ state + d[:, token, ..., None] * u[:, token, :, None, :]
        compressed = (levels * inputs["read"][:, token, None]).sum(-1)
        out = (inputs["rho"][:, token] * compressed).sum(-1) * F.silu(inputs["gamma"][:, token])
        if valid is not None:
            out = out * valid[:, token, None].to(out.dtype)
        outputs.append(out)
        state = A[:, token] @ state + b[:, token, ..., None] * u[:, token, :, None, :]
    return torch.stack(outputs, 1), state


def _emulated_summary(maps, first, last):
    """p1 over tokens [first, last) from a zero carry: (Phi [B,C,K,K], carry [B,C,K,M])."""
    import torch

    A, b, _Cm, _d, u = maps
    batch, _tokens, channels, order, _ = A.shape
    phi = torch.eye(order, dtype=A.dtype).expand(batch, channels, order, order).clone()
    carry = torch.zeros(batch, channels, order, u.shape[-1], dtype=A.dtype)
    for token in range(first, last):
        phi = A[:, token] @ phi
        carry = A[:, token] @ carry + b[:, token, ..., None] * u[:, token, :, None, :]
    return phi, carry


def _emulated_forward(reference, inputs, valid, chunk, arch, exclusive_last=False, state=None):
    """p1 -> tree scan -> p3 at `chunk` tokens; the incoming state of every chunk, and outputs."""
    import torch

    tokens = inputs["x"].shape[1]
    A, b, Cm, d = _token_maps(reference, inputs["alpha"], inputs["beta"], valid, arch)
    u = inputs["x"][..., None] * inputs["write"][:, :, None, :]
    maps = (A, b, Cm, d, u)
    spans = [(first, min(tokens, first + chunk)) for first in range(0, tokens, chunk)]
    if state is None:
        state = torch.zeros(A.shape[0], A.shape[2], A.shape[3], u.shape[-1], dtype=A.dtype)
    summarised = spans[:-1] if exclusive_last else spans
    summaries = [_emulated_summary(maps, first, last) for first, last in summarised]
    if exclusive_last:
        nan = float("nan")
        summaries.append((torch.full_like(summaries[0][0], nan) if summaries else
                          torch.full((A.shape[0], A.shape[2], A.shape[3], A.shape[3]), nan,
                                     dtype=A.dtype),
                          torch.full_like(state, nan)))
    boundaries = _tree_boundaries([phi for phi, _ in summaries], [carry for _, carry in summaries],
                                  state, False, exclusive_last)
    outputs = []
    for (first, last), boundary in zip(spans, boundaries):
        out, state = _emulated_chunk(maps, inputs, valid, first, last, boundary)
        outputs.append(out)
    return torch.cat(outputs, 1), state, maps, spans


def _emulated_backward(reference, inputs, valid, coarse, fine, arch, grad_out):
    """Gradients of sum(output * grad_out) by v2's decomposition: fine checkpoints from p3 at the
    coarse chunks, fine adjoints (b1), their composition through fine Phi^T (reverse tree scan), per-fine-chunk vjp
    (b3)."""
    import torch

    names = ("x", "alpha", "beta", "rho", "gamma", "write", "read")
    tokens = inputs["x"].shape[1]
    with torch.no_grad():
        _out, _final, maps, coarse_spans = _emulated_forward(reference, inputs, valid, coarse, arch)
        A, b, Cm, d, u = maps
        # p3's fine checkpoints: the state at every fine edge, replayed from the coarse boundaries.
        fine_spans = [(first, min(tokens, first + fine)) for first in range(0, tokens, fine)]
        state = torch.zeros(A.shape[0], A.shape[2], A.shape[3], u.shape[-1], dtype=A.dtype)
        checkpoints = []
        for first, last in fine_spans:
            checkpoints.append(state)
            phi, carry = _emulated_summary(maps, first, last)
            state = phi @ state + carry
        fine_phi = [_emulated_summary(maps, first, last)[0] for first, last in fine_spans]

    def chunk_values(first, last, start, leaves):
        sliced = {name: leaves[name] for name in names}
        v = None if valid is None else valid[:, first:last]
        A_, b_, Cm_, d_ = _token_maps(reference, sliced["alpha"], sliced["beta"], v, arch)
        u_ = sliced["x"][..., None] * sliced["write"][:, :, None, :]
        local = {name: leaf for name, leaf in sliced.items()}
        return _emulated_chunk((A_, b_, Cm_, d_, u_), local, v, 0, last - first, start)

    # b1: the adjoint of every fine chunk's opening state with a zero future.
    local = []
    for (first, last), checkpoint in zip(fine_spans, checkpoints):
        start = checkpoint.detach().requires_grad_(True)
        leaves = {name: inputs[name][:, first:last] for name in names}
        out, _ = chunk_values(first, last, start, leaves)
        local.append(torch.autograd.grad((out * grad_out[:, first:last]).sum(), start)[0])
    # the reverse tree scan: adjoint_after(j) = Phi_{j+1}^T adjoint_after(j+1) + local_{j+1}.
    after = _tree_boundaries(fine_phi, local, None, True, False)
    # b3: every fine chunk's vector-Jacobian product.
    grads = {name: torch.zeros_like(inputs[name]) for name in names}
    for (first, last), checkpoint, adjoint in zip(fine_spans, checkpoints, after):
        leaves = {name: inputs[name][:, first:last].detach().requires_grad_(True) for name in names}
        out, final = chunk_values(first, last, checkpoint, leaves)
        total = (out * grad_out[:, first:last]).sum() + (final * adjoint).sum()
        for name, part in zip(names, torch.autograd.grad(total, [leaves[n] for n in names])):
            grads[name][:, first:last] = part
    return grads


def _emulation_inputs(order, memory, tokens, masked, seed, dtype):
    import torch

    generator = torch.Generator().manual_seed(seed)
    draw = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64)
    batch, channels = 1, 3
    inputs = {
        "x": 0.6 * draw(batch, tokens, channels), "alpha": draw(batch, tokens, channels, order),
        "beta": 0.05 * draw(batch, tokens, channels, order),
        "rho": draw(batch, tokens, channels, order + 1) / math.sqrt(order + 1),
        "gamma": draw(batch, tokens, channels), "write": draw(batch, tokens, memory) / math.sqrt(memory),
        "read": draw(batch, tokens, order + 1, memory) / math.sqrt(memory)}
    valid = None
    if masked:
        valid = torch.rand(batch, tokens, generator=generator) > 0.2
        valid[:, 0] = False
    return {name: value.to(dtype) for name, value in inputs.items()}, valid


def check_chunking_cpu(quick=False):
    """v2's chunked computation, at the chunks v2 picks, against the reference, on the CPU."""
    import torch
    from unittest.mock import patch

    cuda_map = importlib.import_module("kernel.cuda.mira_dspv2.mira")
    reference = importlib.import_module("model.mira_mixer")
    arch = "cauchy_x_hyperbolic"
    stage = cuda_map.ARCH_STAGE[arch]
    chosen = set()
    for (sms, optin, per_sm, reserved) in DISPATCH_DEVICES.values():
        attributes = {cuda_map.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: per_sm,
                      cuda_map.CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK: reserved}
        cuda_map._HARDWARE.clear()
        with patch.object(cuda_map.D, "sm_count", return_value=sms), \
                patch.object(cuda_map.D, "max_optin_shared", return_value=optin), \
                patch.object(cuda_map.D, "device_attr",
                             side_effect=lambda index, attr, default=0: attributes.get(attr, default)):
            for order, memory in ((1, 16), (2, 16), (4, 16), (8, 16), (2, 64)):
                for batch, tokens, channels in ((2, 1031, 1536), (19, 512, 1024), (1, 4099, 1536)):
                    if quick and tokens > 1100:
                        continue
                    fine = min(tokens, cuda_map._fine_chunk(
                        None, 0, order, memory, "bf16", batch, tokens, channels, stage)[0])
                    coarse = cuda_map._forward_chunk(
                        None, 0, order, memory, "bf16", batch, tokens, channels, True, fine)
                    inference = cuda_map._inference_chunk(
                        None, 0, order, memory, "bf16", batch, tokens, channels)
                    chosen.add((order, memory, tokens, coarse, fine, inference))
    cuda_map._HARDWARE.clear()
    worst_forward = worst_backward = worst_parity = 0.0
    torch.set_num_threads(1)
    for order, memory, tokens, coarse, fine, inference in sorted(chosen):
        inputs, valid = _emulation_inputs(order, memory, tokens, True, order * 1000 + tokens,
                                          torch.float64)
        sequential, final, _maps, _spans = _emulated_forward(reference, inputs, valid, tokens, arch)
        # the sequential emulation IS the reference's computation (both fp32)
        inputs32 = {name: value.float() for name, value in inputs.items()}
        got32 = reference.matrix_lattice_iir(
            inputs32["x"], inputs32["alpha"], inputs32["rho"], inputs32["gamma"], inputs32["write"],
            inputs32["read"], valid, raw_beta=inputs32["beta"], arch=arch)
        mine32 = _emulated_forward(reference, inputs32, valid, tokens, arch)[0]
        worst_parity = max(worst_parity, _scaled(mine32, got32))
        for chunk, exclusive in ((coarse, False), (inference, True)):
            out, state, _maps, _spans = _emulated_forward(reference, inputs, valid, chunk, arch,
                                                          exclusive_last=exclusive)
            worst_forward = max(worst_forward, _scaled64(out, sequential), _scaled64(state, final))
        grad_out = torch.randn(sequential.shape, generator=torch.Generator().manual_seed(7),
                               dtype=torch.float64)
        leaves = {name: value.clone().requires_grad_(True) for name, value in inputs.items()}
        out, _final, _maps, _spans = _emulated_forward(reference, leaves, valid, tokens, arch)
        (out * grad_out).sum().backward()
        grads = _emulated_backward(reference, inputs, valid, coarse, fine, arch, grad_out)
        for name, value in grads.items():
            worst_backward = max(worst_backward, _scaled64(value, leaves[name].grad))
        print(f"PASS: mira_dspv2 CPU emulation K={order} d_m={memory} T={tokens}: training chunks "
              f"{coarse}/{fine} (coarse/fine), inference chunk {inference}")
    if worst_parity > 1.0e-4 or worst_forward > 1.0e-12 or worst_backward > 1.0e-10:
        raise AssertionError(f"mira_dspv2 CPU emulation: parity {worst_parity:.3e}, forward "
                             f"{worst_forward:.3e}, backward {worst_backward:.3e}")
    print(f"PASS: mira_dspv2 CPU emulation of {len(chosen)} (K, d_m, T, coarse, fine, inference) "
          f"choices: sequential emulation vs matrix_lattice_iir (fp32) {worst_parity:.2e}; "
          f"chunked vs sequential (fp64) forward {worst_forward:.2e}, every gradient "
          f"{worst_backward:.2e}")
    return max(worst_forward, worst_backward)


def _optimized_inference_resources(cuda):
    import torch

    index = torch.cuda.current_device()
    module = cuda.D.module_for(cuda.CUBIN, index, cuda.FNS)
    worst_registers = 0
    failures = []
    for suffix in ("f32", "bf16", "f16"):
        for phase, shared, lanes, group, ceiling, blocks in (
                ("p1", 2048, 2, 192, 80, 2), ("p3", 8192, 1, 128, 96, 5)):
            name = cuda._forward_scan_name(
                phase, 2, 16, lanes, group, suffix, False)
            resources = cuda._kernel_resources(
                module, index, name, lanes * group, shared)
            registers = resources["registers_per_thread"]
            local = resources["local_memory_bytes_per_thread"]
            resident = resources["resident_blocks_per_multiprocessor"]
            print(f"[mira-resources] {name}: registers={registers} local={local} "
                  f"resident={resident} blocks/SM shared={resources['dynamic_shared_memory_bytes']}")
            if registers > ceiling:
                failures.append(
                    f"optimized K=2 inference {name} uses {registers} registers per thread")
            if local:
                failures.append(
                    f"optimized K=2 inference {name} uses {local} local bytes per thread")
            if resident < blocks:
                failures.append(
                    f"optimized K=2 inference {name} has only {resident} resident blocks/SM")
            if resources["dynamic_shared_memory_bytes"] != shared:
                failures.append(
                    f"optimized K=2 inference {name} has wrong dynamic shared memory")
            worst_registers = max(worst_registers, registers)
    if failures:
        raise AssertionError("\n".join(failures))
    return worst_registers


# mira_dspv2: THE COEFFICIENT-FORMING INFERENCE SCAN (fused_scan) AGAINST v3'S PATH AND THE REFERENCE.
def _fused_case(batch, tokens, channels, rank=48, masked=False, with_state=False,
                arch="cauchy_x_mobius", expect_fused=True):
    """matrix_lattice_iir_control against the same coefficients projected by F.linear and run
    through matrix_lattice_iir (v3's path), and against the fp32 reference. The fused path forms
    each coefficient with a different d_r-term summation order than cuBLAS, so a coefficient may
    land one bf16 ulp away; the fallback (expect_fused=False) must be bit-identical."""
    import torch
    import torch.nn.functional as F

    cuda = importlib.import_module("kernel.cuda.mira_dspv2.mira")
    reference = importlib.import_module("model.mira_mixer")
    order, memory, dtype, device = 2, 16, torch.bfloat16, "cuda"
    torch.manual_seed(20261011 + batch + tokens + channels + rank)
    total = channels * (3 * order + 1) + (order + 1) * memory
    spans = ((0, channels * order), (channels * order, 2 * channels * order),
             (2 * channels * order, channels * (3 * order + 1)),
             (channels * (3 * order + 1), total))
    weight = (torch.randn(total, rank, device=device) / math.sqrt(rank)).to(dtype)
    bias = (0.3 * torch.randn(total, device=device)).to(dtype)
    control = torch.nn.functional.silu(torch.randn(batch, tokens, rank, device=device)).to(dtype)
    packed = torch.randn(batch, tokens, 2 * channels + 3, device=device).to(dtype)
    signal, gamma = packed[..., :channels], packed[..., channels:2 * channels]
    write = (torch.randn(batch, tokens, memory, device=device) / 4).to(dtype)
    valid = (torch.rand(batch, tokens, device=device) > 0.25) if masked else None
    state = (torch.randn(batch, order, channels, memory, device=device) if with_state else None)
    with torch.no_grad():
        index = torch.cuda.current_device()
        module = cuda.D.module_for(cuda.CUBIN, index, cuda.FNS)
        plan = cuda._fused_plan(module, index, signal, control, weight, gamma, write, order,
                                memory, cuda.ARCH_STAGE[arch])
        if (plan is not None) != expect_fused:
            raise AssertionError(f"fused plan {plan} at B={batch} T={tokens} C={channels}")
        got = cuda.matrix_lattice_iir_control(
            signal, control, weight, bias, spans, gamma, write, valid=valid, state=state,
            return_state=with_state, arch=arch)
        coefficients = F.linear(control, weight, bias)
        alpha, beta, rho, read = coefficients.split(tuple(hi - lo for lo, hi in spans), dim=-1)
        views = (alpha.view(batch, tokens, channels, order),
                 rho.view(batch, tokens, channels, order + 1),
                 read.view(batch, tokens, order + 1, memory),
                 beta.view(batch, tokens, channels, order))
        want = cuda.matrix_lattice_iir(
            signal, views[0], views[1], gamma, write, views[2], raw_beta=views[3], valid=valid,
            state=state, return_state=with_state, arch=arch)
        exact = reference.matrix_lattice_iir(
            signal.float(), views[0].float(), views[1].float(), gamma.float(), write.float(),
            views[2].float(), valid, raw_beta=views[3].float(), state=state,
            return_state=with_state, arch=arch)
    if with_state:
        (got, got_final), (want, want_final), (exact, exact_final) = got, want, exact
    if not expect_fused:
        if not torch.equal(got, want) or (with_state and not torch.equal(got_final, want_final)):
            raise AssertionError("the unfused fallback must be v3's path bit for bit")
    differing = float((got != want).float().mean())
    error = max(_scaled(got, want), _scaled(got.float(), exact))
    if with_state:
        error = max(error, _scaled(got_final, exact_final))
    if error > 2.0e-2 or differing > 0.02:
        raise AssertionError(
            f"fused scan B={batch} T={tokens} C={channels} d_r={rank} {arch}: scaled error "
            f"{error:.3e}, {differing:.2%} of outputs differ from v3's")
    return error, differing


def check_fused():
    import torch

    if torch.cuda.get_device_capability()[0] < 8:
        print("SKIP: mira_dspv2 fused scan needs compute capability 8.0 (bf16 mma)")
        return 0.0
    worst = 0.0
    for args in ((19, 5, 70, 48, False, False, "cauchy_x_mobius"),
                 (19, 5, 70, 48, True, True, "cauchy_x_mobius"),
                 (3, 4, 32, 16, True, False, "cauchy_x_hyperbolic"),
                 (17, 3, 40, 32, False, True, "cauchy_x_identity"),
                 (5, 16, 64, 64, True, True, "gaussian_x_mobius"),
                 (64, 257, 1536, 48, False, False, "cauchy_x_mobius"),
                 (571, 33, 1536, 48, True, False, "cauchy_x_mobius")):
        error, differing = _fused_case(*args)
        worst = max(worst, error)
        print(f"PASS: mira_dspv2 fused scan B={args[0]} T={args[1]} C={args[2]} d_r={args[3]} "
              f"{args[6]} masked={args[4]} state={args[5]}: error {error:.3e}, "
              f"{differing:.3%} of outputs differ from v3's")
    # mira_dspv2: ONE KERNEL PER STAGE MAP, so every map's kernel is compared (d_r cycling).
    cuda = importlib.import_module("kernel.cuda.mira_dspv2.mira")
    for number, arch in enumerate(cuda.ARCH_STAGE):
        rank = (16, 32, 48, 64)[number % 4]
        error, differing = _fused_case(6 + number, 3 + number % 5, 64 + 32 * (number % 3), rank,
                                       number % 2 == 1, number % 3 == 0, arch)
        worst = max(worst, error)
        print(f"PASS: mira_dspv2 fused scan {arch} d_r={rank}: error {error:.3e}, "
              f"{differing:.3%} of outputs differ from v3's")
    error, _ = _fused_case(1, 300, 1536, 48, expect_fused=False)
    print("PASS: mira_dspv2 split sequence falls back to v3's path bit for bit")
    # THE MIXER: model/mira_mixer.py routes an eval forward through matrix_lattice_iir_control.
    reference = importlib.import_module("model.mira_mixer")
    mixer = reference.MiraMixer(
        768, lattice_order=2, d_e=1536, coefficient_bottleneck=48, memory_dimension=16).cuda()
    mixer = mixer.to(torch.bfloat16).eval()
    # 16 TOKENS IS ONE CHUNK AT ANY BATCH (the chunk is at least the 16-token alignment), so the
    # fused scan serves it; the launches are recorded to prove it.
    x = torch.randn(8, 16, 768, device="cuda", dtype=torch.bfloat16)
    # ONLY THIS PACKAGE'S CUBIN IS NEEDED: the mixer's own pre-v4 path (coefficient_projection
    # materialised, matrix_lattice_iir) is reached by hiding matrix_lattice_iir_control, so the
    # check does not depend on another family having been built.
    from unittest.mock import patch

    cuda = importlib.import_module("kernel.cuda.mira_dspv2.mira")
    previous = reference.MIRA_CUDA
    try:
        reference.use_mira_cuda("mira_dspv2")
        launched = []
        original_launch = cuda.launch

        def record(module, name, *args, **kwargs):
            launched.append(name)
            return original_launch(module, name, *args, **kwargs)

        with torch.no_grad(), patch.object(cuda, "launch", side_effect=record):
            fused = mixer(x)
            if not any(name.startswith(("mira_fused_scan_", "mira_dspv2_fused_scan_"))
                       for name in launched):
                raise AssertionError(f"the mixer did not reach the fused scan: {launched}")
            with patch.object(cuda, "matrix_lattice_iir_control", None):
                plain = mixer(x)
    finally:
        reference.use_mira_cuda(previous)
    error = _scaled(fused, plain)
    if error > 2.0e-2:
        raise AssertionError(f"mixer forward through the fused scan differs by {error:.3e}")
    print(f"PASS: mira_dspv2 mixer eval forward, fused scan vs the materialised path: {error:.3e}")
    return max(worst, error)


def smoke():
    import torch

    check_map()
    if torch.cuda.get_device_capability()[0] >= 8:
        error, _ = _fused_case(19, 5, 70, 48, True, True)
        print(f"PASS: mira_dspv2 fused scan smoke error={error:.3e}")
    initializer = importlib.import_module("model.initializers._common")
    reflection = torch.tensor([[0.2, -0.7], [0.8, -0.4]], dtype=torch.float32)
    _, beta = initializer.stage_coefficients(
        "expsoftplus_x_hyperbolic", reflection, reflection.double(),
        torch.ones_like(reflection), torch.ones_like(reflection))
    radius = torch.exp(-torch.nn.functional.softplus(beta.double()))
    target = 1.0 - (1.0 - reflection.double().abs()) / (1 << 13)
    torch.testing.assert_close(radius, target, rtol=1.0e-7, atol=1.0e-8)
    draw_beta = torch.tensor([0.1, -0.2, 0.3, -0.4], dtype=torch.float32)
    _, beta = initializer.uniform_coefficients(
        "expsoftplus_x_hyperbolic", draw_beta, draw_beta)
    radius = torch.exp(-torch.nn.functional.softplus(beta.double()))
    torch.testing.assert_close(radius, (1.0 + draw_beta.double().square()).reciprocal(),
                               rtol=1.0e-7, atol=1.0e-8)
    _, beta = initializer.stage_coefficients(
        "hill_x_hyperbolic", reflection, reflection.double(),
        torch.ones_like(reflection), torch.ones_like(reflection))
    radius = beta.double().square() / (1.0 + beta.double().square())
    torch.testing.assert_close(radius, target, rtol=1.0e-7, atol=1.0e-8)
    _, beta = initializer.uniform_coefficients("hill_x_hyperbolic", draw_beta, draw_beta)
    radius = beta.double().square() / (1.0 + beta.double().square())
    torch.testing.assert_close(radius, (1.0 + draw_beta.double().square()).reciprocal(),
                               rtol=1.0e-7, atol=1.0e-8)
    if not torch.cuda.is_available():
        print("mira-dsp check: no CUDA device")
        return 1
    cuda = importlib.import_module("kernel.cuda.mira_dspv2.mira")
    ok, detail = cuda.status()
    if not ok:
        print(f"FAILED: {detail}")
        return 1
    cuda.report()
    registers = _optimized_inference_resources(cuda)
    print(f"PASS: optimized K=2 inference resources registers<={registers} local=0 resident p1>=2 p3>=5")
    # THE SMOKE SET IS THE COMPILED SET, READ FROM THE LAUNCHER RATHER THAN RESTATED. The orders
    # were written here as (2, 4) while the launcher compiled ORDERS = (1, 2, 4, 8) and
    # ablation/run.py trained every one of them, so K=1 and K=8 were built, launched for 10,000
    # steps and never once compared against the reference -- and `make quicktest` still printed
    # PASS. Taking both tuples from the launcher means the gate cannot fall behind the cubin again.
    # THE SMOKE SET IS EXACTLY THE SET THAT WAS COMPILED, pair by pair. ORDER_MEMORIES says which
    # memory widths each order is built for -- orders 1 and 8 exist only at d_m 16, because that is
    # the width the ablation runs and building the other two compiled bodies nobody launches -- so
    # iterating orders against every width would name shapes the cubin does not hold.
    pairs = tuple((order, memory) for order in cuda.ORDERS
                  for memory in cuda.ORDER_MEMORIES[order])
    # TWO CASES PER (d_m, K) PAIR AND NOTHING ELSE, so the pre-flight grows only with the compiled
    # set: one bfloat16 shape at T=37, C=35, batch 2, run undamped and then with beta_bias=0.75,
    # which is the only setting under which grad_beta is distinguishable from zero. That is 24
    # cases for the 12 rows of BACKWARD_MAP, up from 12; no dtype or sequence shape is
    # added on top, which is where a pre-flight's runtime would actually go.
    errors = [_case(memory, "bfloat16", 37, 35, order=order)
              for order, memory in pairs]
    errors += [_case(memory, "bfloat16", 37, 35, order=order,
                     beta_bias=0.75)
               for order, memory in pairs]
    errors.append(_case(16, "float32", 37, 35, order=2, beta_bias=-2.0,
                        arch="expsoftplus_x_hyperbolic"))
    errors.append(_packed_case("bfloat16", masked=True, order=2, memory=16,
                               arch="expsoftplus_x_hyperbolic", beta_bias=-2.0))
    # THE OTHER cayley_parameterizations USE THE SAME COMPILED STAGE, so each
    # pair adds one case per arch, at a beta_bias where r is near 0.64 under the cauchy radius
    # (0.75), so grad_beta is not zero.
    for stage_arch in ("cauchy_x_hyperbolic", "cauchy_x_identity", "gaussian_x_mobius",
                       "gaussian_x_hyperbolic", "gaussian_x_identity", "expexp_x_mobius",
                       "expexp_x_hyperbolic", "expexp_x_identity", "expsoftplus_x_mobius",
                       "expsoftplus_x_hyperbolic", "expsoftplus_x_identity",
                       "hill_x_mobius", "hill_x_hyperbolic", "hill_x_identity"):
        errors += [_case(memory, "bfloat16", 37, 35, order=order,
                         beta_bias=DAMPED[stage_arch], arch=stage_arch)
                   for order, memory in pairs]
        print(f"PASS: arch={stage_arch} smoke error={max(errors[-len(pairs):]):.3e}")
    errors.append(max(_packed_case("bfloat16", masked=True, order=order, memory=memory)
                      for order, memory in pairs))
    print(f"PASS: packed masked gradient smoke error={errors[-1]:.3e}")
    errors.append(_state_case(
        2, 16, "bfloat16", 257, 193, (1, 31, 33, 192)))
    errors.append(_state_case(4, 16, "bfloat16"))
    print(f"PASS: cached CUDA smoke error={max(errors[-2:]):.3e}")
    errors.append(max(
        _forward_case(
            16, "bfloat16", 257, 193, 128, order=2, beta_bias=0.75),
        _projection_case("bfloat16")))
    print(f"PASS: optimized K=2 inference smoke error={errors[-1]:.3e}")
    errors.append(_mask_dispatch_case("bfloat16"))
    print(f"PASS: masked/unmasked dispatch smoke error={errors[-1]:.3e}")
    errors.append(max(_fine_case(order, memory, "bfloat16", fine)
                      for order, memory in pairs
                      for fine in _fine_chunks(cuda, order, memory, "bf16")[-1:]))
    print(f"PASS: mira_dspv2 longest backward chunk smoke error={errors[-1]:.3e}")
    print("PASS: MIRA-Matrix "
          + ", ".join(f"K={order} d_m={memory}" for order, memory in pairs)
          + " decoupled shift stage, undamped and damped radius, every cayley_parameterization,"
          + " smoke value/gradient parity; "
          + f"max scaled error {max(errors):.3e}")
    return 0


def main():
    import torch

    check_map()
    if not torch.cuda.is_available():
        print("mira-dsp check: no CUDA device")
        return 1
    cuda = importlib.import_module("kernel.cuda.mira_dspv2.mira")
    ok, detail = cuda.status()
    if not ok:
        print(f"FAILED: {detail}")
        return 1
    cuda.report()
    registers = _optimized_inference_resources(cuda)
    print(f"PASS: optimized K=2 inference resources registers<={registers} local=0 resident p1>=2 p3>=5")
    cases = (
        (16, "float32", 9, 5),
        (16, "bfloat16", 37, 7),
        (32, "float32", 9, 5),
        (32, "bfloat16", 37, 7),
        (32, "float32", 33, 35, 0.75),
        (32, "float32", 33, 35, 3.0),
        (64, "float32", 9, 5),
        (64, "bfloat16", 37, 7),
        (64, "bfloat16", 17, 67),
        (64, "float32", 33, 35, 3.0),
        (64, "float16", 65, 3),
    )
    worst = 0.0
    for case in cases:
        error = _case(*case)
        worst = max(worst, error)
        print(f"PASS: memory={case[0]} dtype={case[1]} "
              f"T={case[2]} C={case[3]} error={error:.3e}")
    for memory in (32, 64):
        for dtype_name in ("float32", "bfloat16"):
            error = _case(memory, dtype_name, 33, 35, order=2)
            worst = max(worst, error)
            print(f"PASS: memory={memory} dtype={dtype_name} "
                  f"K=2 T=33 C=35 error={error:.3e}")
    error = _case(32, "bfloat16", 37, 35, strided=True)
    worst = max(worst, error)
    print(f"PASS: memory=32 dtype=bfloat16 "
          f"T=37 C=35 strided signal/gamma error={error:.3e}")
    for memory in (32, 64):
        error = _forward_case(memory, "bfloat16", 65, 35)
        worst = max(worst, error)
        print(f"PASS: memory={memory} dtype=bfloat16 "
              f"inference T=65 C=35 error={error:.3e}")
        error = _forward_case(memory, "bfloat16", 261, 35, 128)
        worst = max(worst, error)
        print(f"PASS: memory={memory} dtype=bfloat16 "
              f"inference T=261 C=35 chunk=128 error={error:.3e}")
    for dtype_name in ("float32", "bfloat16", "float16"):
        for tokens in (1, 15, 16, 17, 31, 32, 33, 47, 48, 49):
            error = _forward_case(16, dtype_name, tokens, 129, tokens,
                                  order=2, beta_bias=0.75)
            worst = max(worst, error)
            print(f"PASS: K=2 replay memory=16 dtype={dtype_name} "
                  f"T={tokens} C=129 error={error:.3e}")
        for tokens in (35, 37, 39):
            error = _case(16, dtype_name, tokens, 35, order=2, beta_bias=0.75)
            worst = max(worst, error)
            print(f"PASS: K=2 backward memory=16 dtype={dtype_name} "
                  f"T={tokens} C=35 error={error:.3e}")
        error = max(
            _forward_case(
                16, dtype_name, 257, 193, 128, order=2, beta_bias=0.75),
            _forward_case(
                16, dtype_name, 245, 193, 121, order=2, beta_bias=0.75),
            _forward_case(
                16, dtype_name, 485, 193, 241, order=2, beta_bias=0.75),
            _projection_case(dtype_name))
        worst = max(worst, error)
        print(f"PASS: optimized inference memory=16 dtype={dtype_name} "
              f"K=2 T=257 C=193 chunk=128 error={error:.3e}")
        error = _mask_dispatch_case(dtype_name)
        worst = max(worst, error)
        print(f"PASS: masked/unmasked dispatch dtype={dtype_name} error={error:.3e}")
    error = _forward_case(
        32, "float32", 33, 35, alpha_bias=3.0)
    worst = max(worst, error)
    print(f"PASS: memory=32 dtype=float32 "
          f"inference T=33 C=35 alpha_bias=3.0 error={error:.3e}")
    # THE DECOUPLED SHIFTING STAGE HAS TWO COEFFICIENTS, so both orders, both dtypes and both
    # memory widths are run with a radius that is not one: beta_bias moves r off the undamped
    # rotation, which is the only setting under which grad_beta is distinguishable from zero.
    for memory in (16, 64):
        for order in (2, 4):
            for dtype_name in ("float32", "bfloat16"):
                error = _case(memory, dtype_name, 33, 35,
                              order=order, beta_bias=0.75)
                worst = max(worst, error)
                print(f"PASS: decoupled shift stage memory={memory} dtype={dtype_name} "
                      f"K={order} T=33 C=35 error={error:.3e}")
    for memory in (16, 64):
        error = _case(memory, "float32", 37, 7, beta_bias=3.0)
        worst = max(worst, error)
        print(f"PASS: decoupled shift stage memory={memory} dtype=float32 "
              f"K=4 T=37 C=7 beta_bias=3.0 error={error:.3e}")
        error = _case(memory, "bfloat16", 33, 35, alpha_bias=3.0,
                      beta_bias=0.75)
        worst = max(worst, error)
        print(f"PASS: decoupled shift stage memory={memory} dtype=bfloat16 "
              f"K=4 T=33 C=35 alpha_bias=3.0 error={error:.3e}")
        error = _forward_case(memory, "bfloat16", 261, 35, 128,
                              beta_bias=0.75)
        worst = max(worst, error)
        print(f"PASS: decoupled shift stage memory={memory} dtype=bfloat16 "
              f"inference T=261 C=35 chunk=128 error={error:.3e}")
    error = _case(16, "bfloat16", 37, 35, strided=True,
                  beta_bias=0.75)
    worst = max(worst, error)
    print(f"PASS: decoupled shift stage memory=16 dtype=bfloat16 "
          f"T=37 C=35 strided signal/gamma error={error:.3e}")
    error = _forward_case(64, "float32", 65, 35, beta_bias=0.75)
    worst = max(worst, error)
    print(f"PASS: decoupled shift stage memory=64 dtype=float32 "
          f"inference T=65 C=35 error={error:.3e}")
    # THE READOUT SiLU IS A SECOND ARM OF THE SAME KERNEL, so it is checked at both orders, both
    # dtypes and both memory widths against the reference that carries it, in value and in every
    # gradient, and once in inference where p3 is given a null sum buffer.
    for memory in (16, 64):
        for order in (2, 4):
            for dtype_name in ("float32", "bfloat16"):
                error = _case(memory, dtype_name, 33, 35, order=order,
                              beta_bias=0.75)
                worst = max(worst, error)
                print(f"PASS: readout SiLU memory={memory} dtype={dtype_name} "
                      f"K={order} T=33 C=35 error={error:.3e}")
    for memory in (16, 64):
        error = _forward_case(memory, "bfloat16", 261, 35, 128,
                              beta_bias=0.75)
        worst = max(worst, error)
        print(f"PASS: readout SiLU memory={memory} dtype=bfloat16 "
              f"inference T=261 C=35 chunk=128 error={error:.3e}")
    for dtype_name in ("float32", "bfloat16", "float16"):
        for masked in (False, True):
            error = max(_packed_case(dtype_name, masked, order=order, memory=memory)
                        for order, memory in cuda._built_pairs()
                        )
            worst = max(worst, error)
            print(f"PASS: packed projections dtype={dtype_name} masked={masked} error={error:.3e}")
    for order, memory in cuda._built_pairs():
        error = _state_case(order, memory, "float32")
        worst = max(worst, error)
        print(f"PASS: masked streaming K={order} memory={memory} error={error:.3e}")
    for dtype_name in ("bfloat16", "float16"):
        error = max(
            _state_case(2, 16, dtype_name, 257, 193, (1, 31, 33, 192)),
            _state_case(2, 16, dtype_name, 31, 193, (15, 16)),
            _state_case(4, 16, dtype_name))
        worst = max(worst, error)
        print(f"PASS: masked streaming dtype={dtype_name} error={error:.3e}")
    # THE TREE SCAN: every built (K, d_m), chunk counts across warp and block edges, the worker
    # count the plan picks and every forced power of two up to 256, forward (with an initial state),
    # backward, and inference with the last summary poisoned by NaN.
    for order, memory in cuda._built_pairs():
        error = 0.0
        for chunks in (1, 2, 17, 33, 127, 128, 129, 2051):
            for workers in (None, 1, 2, 4, 32, 64, 256):
                if workers is not None and workers > max(1, 1 << (chunks - 1).bit_length()):
                    continue
                for reverse, inference in ((False, False), (True, False), (False, True)):
                    error = max(error, _boundary_case(
                        chunks, reverse, inference=inference, poison_final=inference,
                        workers=workers, order=order, memory=memory, batch=2, channels=3))
        worst = max(worst, error)
        print(f"PASS: tree-scan boundaries K={order} d_m={memory} error={error:.3e}")
    error = max(_packed_case(dtype_name, order=order, memory=memory,
                              batch=256, tokens=257, channels=5)
                for order, memory in cuda._built_pairs()
                for dtype_name in ("float32", "bfloat16", "float16"))
    worst = max(worst, error)
    print(f"PASS: coarse forward/fine backward checkpoints error={error:.3e}")
    error = _case(16, "float32", 3, 3, beta_bias=2e19)
    worst = max(worst, error)
    print(f"PASS: finite large damping gradient error={error:.3e}")
    error = _read_only_gradient_case()
    worst = max(worst, error)
    print(f"PASS: memory-read-only gradient error={error:.3e}")
    # THE OTHER cayley_parameterizations use the same compiled stage with mobius or hyperbolic rotation.
    for stage_arch in ("cauchy_x_hyperbolic", "cauchy_x_identity", "gaussian_x_mobius",
                       "gaussian_x_hyperbolic", "gaussian_x_identity", "expexp_x_mobius",
                       "expexp_x_hyperbolic", "expexp_x_identity", "expsoftplus_x_mobius",
                       "expsoftplus_x_hyperbolic", "expsoftplus_x_identity",
                       "hill_x_mobius", "hill_x_hyperbolic", "hill_x_identity"):
        for memory in (16, 64):
            for order in (2, 4):
                for dtype_name in ("float32", "bfloat16"):
                    for beta_bias in (-3.0, 0.75):
                        error = _case(memory, dtype_name, 33, 35, order=order,
                                      beta_bias=beta_bias, arch=stage_arch)
                        worst = max(worst, error)
                        print(f"PASS: arch={stage_arch} memory={memory} dtype={dtype_name} "
                              f"K={order} T=33 C=35 beta_bias={beta_bias} error={error:.3e}")
            error = _case(memory, "bfloat16", 33, 35, alpha_bias=3.0,
                          beta_bias=DAMPED[stage_arch], arch=stage_arch)
            worst = max(worst, error)
            print(f"PASS: arch={stage_arch} memory={memory} dtype=bfloat16 "
                  f"K=4 T=33 C=35 alpha_bias=3.0 error={error:.3e}")
            error = _forward_case(memory, "bfloat16", 261, 35, 128,
                                  beta_bias=DAMPED[stage_arch], arch=stage_arch)
            worst = max(worst, error)
            print(f"PASS: arch={stage_arch} memory={memory} dtype=bfloat16 "
                  f"inference T=261 C=35 chunk=128 error={error:.3e}")
        for dtype_name in ("float32", "bfloat16"):
            error = _forward_case(16, dtype_name, 257, 193, 128, order=2,
                                  beta_bias=DAMPED[stage_arch], arch=stage_arch)
            worst = max(worst, error)
            print(f"PASS: arch={stage_arch} optimized inference memory=16 dtype={dtype_name} "
                  f"K=2 T=257 C=193 chunk=128 error={error:.3e}")
            for masked in (False, True):
                error = max(_packed_case(dtype_name, masked, order=order, memory=memory,
                                         arch=stage_arch, beta_bias=DAMPED[stage_arch])
                            for order, memory in cuda._built_pairs())
                worst = max(worst, error)
                print(f"PASS: arch={stage_arch} packed projections dtype={dtype_name} "
                      f"masked={masked} error={error:.3e}")
        for order, memory in cuda._built_pairs():
            error = _state_case(order, memory, "float32", arch=stage_arch,
                                beta_bias=DAMPED[stage_arch])
            worst = max(worst, error)
            print(f"PASS: arch={stage_arch} masked streaming K={order} memory={memory} "
                  f"error={error:.3e}")
        error = _case(16, "float32", 3, 3, beta_bias=100.0, arch=stage_arch)
        worst = max(worst, error)
        print(f"PASS: arch={stage_arch} finite large damping gradient error={error:.3e}")
        if stage_arch.endswith("_x_identity"):
            # theta = alpha is periodic and unbounded: |alpha| near 100 exercises the 2 pi
            # argument reduction ahead of __sincosf.
            error = _case(16, "float32", 33, 35, order=2, alpha_bias=100.0,
                          beta_bias=DAMPED[stage_arch], arch=stage_arch)
            worst = max(worst, error)
            print(f"PASS: arch={stage_arch} alpha_bias=100.0 error={error:.3e}")
        if stage_arch.startswith("hill_x_"):
            error = _case(16, "float32", 3, 3, order=2, beta_bias=0.0, arch=stage_arch)
            worst = max(worst, error)
            print(f"PASS: arch={stage_arch} beta_bias=0.0 gradient error={error:.3e}")
    for order, memory in cuda._built_pairs():
        for dtype_name, suffix in (("float32", "f32"), ("bfloat16", "bf16")):
            for fine in _fine_chunks(cuda, order, memory, suffix):
                error = _fine_case(order, memory, dtype_name, fine)
                worst = max(worst, error)
                print(f"PASS: mira_dspv2 backward chunk {fine} K={order} memory={memory} "
                      f"dtype={dtype_name} error={error:.3e}")
    for tokens in (257, 1031, 4099):
        error = _forward_case(16, "bfloat16", tokens, 193, order=2, beta_bias=0.75)
        worst = max(worst, error)
        print(f"PASS: mira_dspv2 dispatched inference K=2 memory=16 T={tokens} error={error:.3e}")
    worst = max(worst, check_fused())
    print(f"PASS: all MIRA-Matrix v7 values and gradients; worst scaled error {worst:.3e}")
    return 0


def cpu_main(quick=False):
    """Everything that needs no GPU: the dispatch rules on every GPU of DISPATCH_DEVICES, and the
    chunked computation at the chunks they pick, against the reference.

        python kernel/cuda/mira_dspv2/check.py --cpu"""
    check_map()
    check_tree_cpu(quick=quick)
    check_chunking_cpu(quick=quick)
    print("PASS: mira_dspv2 CPU checks")
    return 0


TOLERANCE = 3.0e-2
MEAN_TOLERANCE = 2.0e-3


def _mixer(width, d_e, rank, memory, arch, device, seed):
    import torch

    from model.mira_mixer import MiraMixer

    torch.manual_seed(seed)
    mixer = MiraMixer(width, lattice_order=2, d_e=d_e, coefficient_bottleneck=rank,
                      memory_dimension=memory, arch=arch)
    return mixer.to(device=device, dtype=torch.bfloat16).eval()


def _inputs(mixer, z):
    import torch.nn.functional as F

    tokens = z.shape[1]
    projected, control, gamma = mixer.project_recurrence_inputs(z, tokens, coefficients=False)
    signal, raw_write = projected.split((mixer.inner, mixer.memory_dimension), dim=-1)
    return signal, control, gamma, F.silu(raw_write).contiguous()


def _spans(mixer):
    return (mixer.alpha_span, mixer.beta_span, mixer.rho_span, mixer.read_span)


def _compare(label, got, want):
    import torch

    got, want = got.float(), want.float()
    scale = want.abs().max().clamp_min(1.0e-6)
    worst = float((got - want).abs().max() / scale)
    mean = float((got - want).abs().mean() / scale)
    ok = bool(torch.isfinite(got).all()) and worst < TOLERANCE and mean < MEAN_TOLERANCE
    print(f"{'PASS' if ok else 'MISMATCH'}: {label} max {worst:.3e} mean {mean:.3e} "
          f"(of the reference's largest value)")
    return ok


def _torch_reference(mixer, signal, control, gamma, memory_write, valid=None, state=None,
                     return_state=False):
    import torch.nn.functional as F

    from model.mira_mixer import matrix_lattice_iir

    batch, tokens, channels = signal.shape
    order = mixer.order
    projection = mixer.coefficient_projection
    coefficients = F.linear(control, projection.weight, projection.bias)
    alpha, beta, rho, memory_read = coefficients.split(
        tuple(hi - lo for lo, hi in _spans(mixer)), dim=-1)
    return matrix_lattice_iir(
        signal, alpha.reshape(batch, tokens, channels, order),
        rho.reshape(batch, tokens, channels, order + 1), gamma, memory_write,
        memory_read.reshape(batch, tokens, order + 1, -1), valid=valid,
        raw_beta=beta.reshape(batch, tokens, channels, order), state=state,
        return_state=return_state, arch=mixer.arch)


def _emulated(mixer, signal, control, gamma, memory_write, valid=None, state=None,
              kind="sa"):
    import numpy as np
    import torch
    import torch.nn.functional as F

    from kernel.cuda.mira_dspv2 import mira as sa
    from simulator.nvidia import emulate, gpu, sass_parse, sm

    batch, tokens, channels = signal.shape
    signal, gamma, control = signal.contiguous(), gamma.contiguous(), control.contiguous()
    order, memory = mixer.order, mixer.memory_dimension
    projection = mixer.coefficient_projection
    weights, bias = sa._packed_coefficient_weights(
        projection.weight, projection.bias, _spans(mixer)[:3], order, channels)
    output = torch.zeros(batch, tokens, channels, dtype=torch.bfloat16)
    final = torch.zeros(batch, order, channels, memory, dtype=torch.float32)
    arrays = {}

    def place(name, tensor):
        data = tensor.detach().contiguous().view(torch.uint8).numpy().reshape(-1).copy()
        arrays[name] = ((len(arrays) + 1) << 36, data)
        return arrays[name][0]

    stage = sa.ARCH_STAGE[mixer.arch]
    inputs = [place("x", signal), place("control", control), place("weight", weights),
              place("bias", bias), place("gamma", gamma), place("memory_write", memory_write)]
    tail = [0 if state is None else place("initial", state), place("final", final),
            0 if valid is None else place("valid", valid)]
    if kind == "sa":
        read_weight, read_bias = sa._read_weights(projection.weight, projection.bias,
                                                  mixer.read_span)
        pointers = inputs + [place("read_weight", read_weight), place("read_bias", read_bias),
                             place("output", output)] + tail
        ints = [batch, tokens, channels, signal.stride(1), gamma.stride(1), control.shape[-1],
                stage]
        threads, rows = sa.SA_THREADS, sa.SA_BATCH
    else:
        lo, hi = mixer.read_span
        memory_read = F.linear(control, projection.weight[lo:hi],
                               None if projection.bias is None else projection.bias[lo:hi])
        pointers = inputs + [place("memory_read", memory_read.contiguous()),
                             place("output", output)] + tail + [0]
        ints = [batch, tokens, channels, signal.stride(1), gamma.stride(1), control.shape[-1],
                (order + 1) * memory, stage]
        threads, rows = sa.FUSED_THREADS, sa.FUSED_BATCH
    groups = (channels + sa.SA_CHANNELS - 1) // sa.SA_CHANNELS
    blocks = ((batch + rows - 1) // rows) * groups
    launch = emulate.Launch(blocks, threads, [("pointer", p) for p in pointers]
                            + [("int", v) for v in ints], memory=arrays)
    program = sass_parse.parse(os.path.join(os.path.dirname(gpu.__file__), "sass",
                                            gpu.KERNELS["mira_dspv2"]["sass"] if kind == "sa"
                                            else "mira_dsp_r48_s6_sm89.sass"))
    head = gpu.tile_loop(program)
    sim = emulate.ExecSim(program, launch, list(range(blocks)), sm.Params(), None,
                          lambda index: "consumer", {"consumer": head}, threads // 32,
                          data=True).run()
    if not sim.finished():
        raise RuntimeError(f"mira_dspv2 {kind}: the emulated kernel did not finish")

    def read(name, like, dtype):
        raw = arrays[name][1].view(np.int16 if dtype is torch.bfloat16 else np.float32).copy()
        return torch.from_numpy(raw).view(dtype).view(like.shape)

    return read("output", output, torch.bfloat16), read("final", final, torch.float32)


def emulate_sass(seed=0):
    import torch

    passed = True
    for batch, tokens, width, d_e in ((9, 26, 64, 64), (19, 22, 64, 40)):
        mixer = _mixer(width, d_e, 48, 16, "cauchy_x_hyperbolic", "cpu", seed)
        torch.manual_seed(seed + batch)
        z = torch.randn(batch, tokens, width, dtype=torch.bfloat16)
        shape = f"B={batch} T={tokens} d={width} d_e={d_e} d_r=48"
        with torch.no_grad():
            signal, control, gamma, memory_write = _inputs(mixer, z)
            valid = torch.rand(batch, tokens) > 0.25
            state = torch.randn(batch, mixer.order, mixer.inner, mixer.memory_dimension) * 0.1
            want = _torch_reference(mixer, signal, control, gamma, memory_write)
            want_masked, want_state = _torch_reference(
                mixer, signal, control, gamma, memory_write, valid=valid, state=state,
                return_state=True)
            runs = {}
            for kind in ("sa", "fused"):
                plain, _ = _emulated(mixer, signal, control, gamma, memory_write, kind=kind)
                masked, final = _emulated(mixer, signal, control, gamma, memory_write,
                                          valid=valid, state=state, kind=kind)
                runs[kind] = (plain, masked, final)
            got, got_masked, got_state = runs["sa"]
            dsp_plain, dsp_masked, dsp_state = runs["fused"]
            for reference, (plain, masked, final) in (
                    ("the PyTorch reference", (want, want_masked, want_state)),
                    ("the 2-token fused scan's emulated SASS", (dsp_plain, dsp_masked, dsp_state))):
                label = f"mira_dspv2's emulated sm_89 SASS against {reference}, {shape}"
                passed &= _compare(f"{label}: output", got, plain)
                passed &= _compare(f"{label}: masked output from a carried state",
                                   got_masked, masked)
                passed &= _compare(f"{label}: final state", got_state, final)
    print("PASS: the emulated mira_dspv2 SASS agrees with the PyTorch reference and the 2-token fused scan"
          if passed else
          "MISMATCH: the emulated mira_dspv2 SASS disagrees with the PyTorch reference or the 2-token fused scan")
    return 0 if passed else 1


if __name__ == "__main__":
    if "--cpu" in sys.argv[1:]:
        raise SystemExit(cpu_main(quick="--quick" in sys.argv[1:]))
    if "--emulate" in sys.argv[1:]:
        raise SystemExit(emulate_sass())
    raise SystemExit(main())
