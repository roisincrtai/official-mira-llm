from pathlib import Path
import importlib
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import torch_npu
from tqdm import tqdm

from model.mira_mixer import matrix_lattice_iir

backend = importlib.import_module("kernel.ascend.910_series.ascend")
ARCHES = backend.ARCHES
matrix_lattice_iir = backend.matrix_lattice_iir
status = backend.status


def check(arch, order, memory, dtype, tokens=7, channels=3):
    batch = 2
    shapes = ((batch, tokens, channels), (batch, tokens, channels, order),
              (batch, tokens, channels, order + 1), (batch, tokens, channels),
              (batch, tokens, memory), (batch, tokens, order + 1, memory),
              (batch, tokens, channels, order))
    reference = [(torch.empty(shape).uniform_(-0.4, 0.4).to(dtype)).requires_grad_()
                 for shape in shapes]
    inputs = [value.detach().to("npu").requires_grad_() for value in reference]
    valid = torch.tensor([[token % 3 != 1 for token in range(tokens)], [False] * tokens])
    expected = matrix_lattice_iir(*reference[:6], raw_beta=reference[6], valid=valid, arch=arch)
    actual = matrix_lattice_iir(*inputs[:6], raw_beta=inputs[6], valid=valid.to("npu"), arch=arch)
    tolerance = {torch.float32: 2e-5, torch.float16: 3e-3, torch.bfloat16: 3e-2}[dtype]
    torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)
    gradient = torch.empty(expected.shape).uniform_(-0.5, 0.5).to(dtype)
    expected.backward(gradient)
    actual.backward(gradient.to("npu"))
    for value, result in zip(reference, inputs):
        assert result.grad is not None and torch.isfinite(result.grad).all()
        torch.testing.assert_close(result.grad.cpu(), value.grad, atol=tolerance, rtol=tolerance)
    initial = torch.empty(batch, order, channels, memory).uniform_(-0.1, 0.1)
    with torch.no_grad():
        expected, expected_state = matrix_lattice_iir(
            *reference[:6], raw_beta=reference[6], valid=valid, state=initial,
            return_state=True, arch=arch)
        actual, actual_state = matrix_lattice_iir(
            *inputs[:6], raw_beta=inputs[6], valid=valid.to("npu"), state=initial.to("npu"),
            return_state=True, arch=arch)
        torch.testing.assert_close(actual.cpu(), expected, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(actual_state.cpu(), expected_state, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(actual_state[1].cpu(), initial[1], atol=0, rtol=0)
        if tokens > 1:
            split = tokens // 2
            first, middle = matrix_lattice_iir(
                *(value[:, :split] for value in inputs[:6]), raw_beta=inputs[6][:, :split],
                valid=valid[:, :split].to("npu"), state=initial.to("npu"), return_state=True, arch=arch)
            second, final = matrix_lattice_iir(
                *(value[:, split:] for value in inputs[:6]), raw_beta=inputs[6][:, split:],
                valid=valid[:, split:].to("npu"), state=middle, return_state=True, arch=arch)
            torch.testing.assert_close(torch.cat((first, second), dim=1), actual,
                                       atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(final, actual_state, atol=tolerance, rtol=tolerance)


def main():
    ready, reason = status()
    if not ready:
        raise RuntimeError(reason)
    torch.manual_seed(0)
    dtypes = [torch.float32, torch.float16]
    if torch_npu.npu.is_bf16_supported():
        dtypes.append(torch.bfloat16)
    cases = [(arch, 2, 16, dtype) for arch in ARCHES for dtype in dtypes]
    cases += [("cauchy_x_hyperbolic", order, memory, torch.float32)
              for order, memory in ((1, 8), (4, 32), (8, 64))]
    cases += [("cauchy_x_hyperbolic", 2, 8, torch.float32, 1, 129),
              ("cauchy_x_hyperbolic", 2, 16, torch.float32, 33, 1)]
    with tqdm(cases, desc="verifying Ascend lattice", unit="case") as progress:
        for completed, case in enumerate(progress, 1):
            check(*case)
            progress.set_postfix(passed=completed)
    print(f"Ascend lattice: {len(cases)} forward, backward, mask and cache cases passed")


if __name__ == "__main__":
    main()
