import os
import sys

import torch


def npu_available():
    try:
        import torch_npu
    except ImportError:
        return False
    return bool(torch_npu.npu.is_available())


def resolve(choice="auto"):
    choice = str(choice)
    if choice in ("auto", "cuda"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if npu_available():
            return torch.device("npu")
        if choice == "cuda":
            raise RuntimeError("no CUDA or Ascend accelerator is available")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if choice.startswith("npu") and not npu_available():
        raise RuntimeError("the selected Ascend NPU is unavailable")
    return torch.device(choice)


def accelerator(device=None):
    kind = resolve(os.environ.get("GPU", "auto")).type if device is None else torch.device(device).type
    if kind == "cuda":
        return torch.cuda
    if kind == "npu":
        if not npu_available():
            raise RuntimeError("Ascend NPU is unavailable")
        return torch.npu
    return None


def count(choice="auto"):
    backend = accelerator(resolve(choice))
    return backend.device_count() if backend is not None else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "count":
        print(count(sys.argv[2] if len(sys.argv) > 2 else "auto"))
    else:
        print(resolve(sys.argv[1] if len(sys.argv) > 1 else "auto").type)
