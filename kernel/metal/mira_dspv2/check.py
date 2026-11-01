from __future__ import annotations

import argparse
import importlib
import itertools
import os
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
while ROOT != os.path.dirname(ROOT) and not os.path.isdir(os.path.join(ROOT, "model")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)


DAMPED = {"cauchy": 0.75, "gaussian": 0.67, "expexp": -0.81, "expsoftplus": -0.58, "hill": 1.33}
NAMES = ("out", "dx", "dalpha", "dbeta", "drho", "dgamma", "dwrite", "dread")
TOLERANCE = {"float32": 2.0e-5, "float16": 1.5e-2, "bfloat16": 3.0e-2}


def _scaled(got, want):
    got = got.detach().cpu().double()
    want = want.detach().cpu().double()
    scale = float(want.abs().max())
    error = float((got - want).abs().max())
    return error / scale if scale > 0.0 else error


def _inputs(batch, tokens, channels, order, memory, arch, seed, alpha_bias=0.0):
    import torch

    generator = torch.Generator().manual_seed(seed)
    shapes = ((batch, tokens, channels), (batch, tokens, channels, order),
              (batch, tokens, channels, order), (batch, tokens, channels, order + 1),
              (batch, tokens, channels), (batch, tokens, memory),
              (batch, tokens, order + 1, memory))
    scales = (0.5, 0.6, 0.3, 0.5, 0.8, 0.5, 0.5)
    values = [torch.randn(*shape, generator=generator) * scale
              for shape, scale in zip(shapes, scales)]
    values[1] += alpha_bias
    values[2] += DAMPED[arch.split("_x_")[0]]
    return values


def _packed(leaves):
    import torch

    batch, tokens, channels = leaves[0].shape
    order = leaves[1].shape[-1]
    memory = leaves[5].shape[-1]
    widths = (channels, channels * order, channels * order, channels * (order + 1), channels,
              memory, (order + 1) * memory)
    storage = torch.cat([leaf.reshape(batch, tokens, -1) for leaf in leaves], dim=-1)
    pieces = storage.split(widths, dim=-1)
    return [pieces[0], pieces[1].view(batch, tokens, channels, order),
            pieces[2].view(batch, tokens, channels, order),
            pieces[3].view(batch, tokens, channels, order + 1), pieces[4], pieces[5],
            pieces[6].view(batch, tokens, order + 1, memory)]


def _compare(metal, reference, values, arch, dtype, valid=None, strided=False):
    import torch

    device = torch.device("mps")
    rounded = [value.to(dtype).float().detach().clone() for value in values]
    upstream = torch.randn(values[0].shape, generator=torch.Generator().manual_seed(7))
    leaves = [value.to(device=device, dtype=dtype).detach().clone().requires_grad_(True)
              for value in rounded]
    inputs = _packed(leaves) if strided else leaves
    x, alpha, beta, rho, gamma, write, read = inputs
    mask = None if valid is None else valid.to(device)
    output = metal.matrix_lattice_iir(x, alpha, rho, gamma, write, read, raw_beta=beta,
                                      valid=mask, arch=arch)
    if output is None:
        raise RuntimeError(f"mira_dspv2 Metal declined {arch} {tuple(values[1].shape)} {dtype}")
    (output.float() * upstream.to(device)).sum().backward()
    want_leaves = [value.detach().clone().requires_grad_(True) for value in rounded]
    x, alpha, beta, rho, gamma, write, read = want_leaves
    want = reference.matrix_lattice_iir(x, alpha, rho, gamma, write, read, valid=valid,
                                        raw_beta=beta, arch=arch)
    (want * upstream).sum().backward()
    torch.mps.synchronize()
    return ([_scaled(output, want)]
            + [_scaled(leaf.grad, target.grad) for leaf, target in zip(leaves, want_leaves)])


def _carried(metal, reference, values, arch, cut):
    import torch

    device = torch.device("mps")
    x, alpha, beta, rho, gamma, write, read = values
    valid = torch.rand(x.shape[:2], generator=torch.Generator().manual_seed(3)) > 0.2
    with torch.no_grad():
        whole, whole_state = reference.matrix_lattice_iir(
            x, alpha, rho, gamma, write, read, valid=valid, raw_beta=beta, return_state=True,
            arch=arch)
        moved = [value.to(device) for value in values]
        x, alpha, beta, rho, gamma, write, read = moved
        mask = valid.to(device)
        first, state = metal.matrix_lattice_iir(
            x[:, :cut], alpha[:, :cut], rho[:, :cut], gamma[:, :cut], write[:, :cut],
            read[:, :cut], raw_beta=beta[:, :cut], valid=mask[:, :cut], return_state=True,
            arch=arch)
        second, final = metal.matrix_lattice_iir(
            x[:, cut:], alpha[:, cut:], rho[:, cut:], gamma[:, cut:], write[:, cut:],
            read[:, cut:], raw_beta=beta[:, cut:], valid=mask[:, cut:], state=state,
            return_state=True, arch=arch)
        plain = metal.matrix_lattice_iir(x, alpha, rho, gamma, write, read, raw_beta=beta,
                                         valid=mask, arch=arch)
    torch.mps.synchronize()
    return [_scaled(torch.cat((first, second), 1), whole), _scaled(final, whole_state),
            _scaled(plain, whole)]


def _control(metal, reference):
    import torch
    import torch.nn.functional as F

    device = torch.device("mps")
    generator = torch.Generator().manual_seed(11)
    channels, order, memory, rank = 9, 2, 16, 24
    spans, start = [], 0
    for width in (channels * order, channels * order, channels * (order + 1),
                  (order + 1) * memory):
        spans.append((start, start + width))
        start += width
    weight = torch.randn(start, rank, generator=generator) * 0.2
    bias = torch.randn(start, generator=generator) * 0.1
    control = torch.randn(2, 13, rank, generator=generator)
    x = torch.randn(2, 13, channels, generator=generator) * 0.5
    gamma = torch.randn(2, 13, channels, generator=generator)
    write = torch.randn(2, 13, memory, generator=generator) * 0.5
    with torch.no_grad():
        got = metal.matrix_lattice_iir_control(
            x.to(device), control.to(device), weight.to(device), bias.to(device), spans,
            gamma.to(device), write.to(device), arch="cauchy_x_hyperbolic")
        alpha, beta, rho, read = F.linear(control, weight, bias).split(
            [hi - lo for lo, hi in spans], dim=-1)
        want = reference.matrix_lattice_iir(
            x, alpha.reshape(2, 13, channels, order), rho.reshape(2, 13, channels, order + 1),
            gamma, write, read.reshape(2, 13, order + 1, memory),
            raw_beta=beta.reshape(2, 13, channels, order))
    torch.mps.synchronize()
    return [_scaled(got, want)]


def run(quick=False):
    import torch

    metal = importlib.import_module("kernel.metal.mira_dspv2.mira")
    reference = importlib.import_module("model.mira_mixer")
    ok, where = metal.status()
    if not ok:
        print(f"[mira_dspv2 metal] unavailable: {where}", flush=True)
        return 1
    print(f"[mira_dspv2 metal] {where}", flush=True)
    failures = []

    def note(label, errors, tolerance, names=NAMES):
        bad = max(errors) >= tolerance
        if bad:
            failures.append(label)
        print(f"{label:54s} " + " ".join(f"{name}={error:.1e}" for name, error in
                                          zip(names, errors)) + ("  FAIL" if bad else ""),
              flush=True)

    archs = [f"{radius}_x_{rotation}" for radius in metal.RADII for rotation in metal.ROTATIONS]
    if quick:
        archs = ["cauchy_x_hyperbolic", "gaussian_x_identity", "hill_x_mobius"]
    for index, arch in enumerate(archs):
        for bias in ((0.0,) if quick else (0.0, 1.7)):
            values = _inputs(2, 37, 21, 2, 16, arch, 100 + index, alpha_bias=bias)
            valid = torch.rand(2, 37, generator=torch.Generator().manual_seed(index)) > 0.25
            note(f"{arch} K2 d_m16 alpha+{bias} masked",
                 _compare(metal, reference, values, arch, torch.float32, valid=valid),
                 TOLERANCE["float32"])
    pairs = ([(2, 16), (4, 64)] if quick else
             [(order, memory) for order in metal.ORDERS for memory in metal.ORDER_MEMORIES[order]])
    for order, memory in pairs:
        values = _inputs(2, 70, 13, order, memory, "cauchy_x_hyperbolic", order * 100 + memory)
        note(f"cauchy_x_hyperbolic K{order} d_m{memory}",
             _compare(metal, reference, values, "cauchy_x_hyperbolic", torch.float32),
             TOLERANCE["float32"])
    values = _inputs(3, 41, 11, 4, 16, "hill_x_mobius", 5)
    note("hill_x_mobius K4 d_m16 packed views",
         _compare(metal, reference, values, "hill_x_mobius", torch.float32, strided=True),
         TOLERANCE["float32"])
    for name in (("bfloat16",) if quick else ("bfloat16", "float16")):
        dtype = getattr(torch, name)
        values = _inputs(2, 45, 40, 2, 16, "cauchy_x_hyperbolic", 21)
        if not metal._ready(2, 16, dtype):
            print(f"{name}: not served on this Mac", flush=True)
            continue
        note(f"cauchy_x_hyperbolic K2 d_m16 {name}",
             _compare(metal, reference, values, "cauchy_x_hyperbolic", dtype),
             TOLERANCE[name])
    for arch in (("cauchy_x_hyperbolic",) if quick else
                 ("cauchy_x_hyperbolic", "gaussian_x_identity", "expsoftplus_x_mobius")):
        values = _inputs(2, 50, 6, 2, 16, arch, 77)
        note(f"{arch} state carried over two calls", _carried(metal, reference, values, arch, 23),
             TOLERANCE["float32"], names=("joined", "final", "whole"))
    note("matrix_lattice_iir_control", _control(metal, reference), TOLERANCE["float32"])
    print(f"[mira_dspv2 metal] {'all passed' if not failures else 'failed: ' + ', '.join(failures)}",
          flush=True)
    return 1 if failures else 0


def smoke():
    return run(quick=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    sys.exit(run(quick=parser.parse_args().quick))
