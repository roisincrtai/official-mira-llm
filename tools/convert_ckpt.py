"""tools/convert_ckpt.py -- drop `arch` and `pe` from every MIRA checkpoint (2026-10-13).

    python3 -m tools.convert_ckpt                  checkpoints/ and saved_models/
    python3 -m tools.convert_ckpt <path> [...]     these files or directories only
    python3 -m tools.convert_ckpt --dry-run        report what would change, change nothing

MiraLLM has no `arch` and no `pe`: it records its lattice maps as lattice_modulus_map and
lattice_argument_map. Every MIRA model cfg in a checkpoint (.pt, .pth, .ckpt) or a config.json loses
its `arch` and `pe` keys; lattice_modulus_map and lattice_argument_map are kept as they are. A cfg
that records the maps only through `arch` (e.g. "arch": "cauchy_x_hyperbolic", or a former name
such as "mira-lattice", read by default_config.lattice_arch) first gets lattice_modulus_map and
lattice_argument_map written from it, so it keeps the maps it was trained with. A cfg whose `arch`
names no lattice map and that records no maps is reported and left unchanged.

ONLY MIRA'S OWN CHECKPOINTS ARE TOUCHED. A run directory is MIRA's when its config.json is a MIRA
model cfg (it holds lattice_order); only the config.json and the checkpoint files (.pt, .pth, .ckpt)
directly in such a directory are opened and converted. Every other directory -- mha, mamba, rwkv
and the other families, whose `arch` names the family -- is skipped after reading its config.json,
and no file in it is opened or written. lattice_order is only read, never changed.

Each changed file is written to a temporary file beside it, read back and checked, then renamed
over the original. Running it again converts nothing.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import default_config as config                      # noqa: E402

CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt")
SKIPPED_KEYS = ("model", "optimizer", "opt", "state_dict")
DROPPED_KEYS = ("arch", "pe")


def is_mira_cfg(value):
    return isinstance(value, dict) and "lattice_order" in value


class Unconvertible(ValueError):
    """A MIRA cfg whose lattice maps are neither recorded nor named by its arch."""


def convert_cfg(cfg):
    """Drop arch and pe from one MIRA cfg, writing lattice_modulus_map and lattice_argument_map from
    arch first when the cfg has neither. True when it changed."""
    present = [key for key in DROPPED_KEYS if key in cfg]
    if not present:
        return False
    if "lattice_modulus_map" not in cfg or "lattice_argument_map" not in cfg:
        lattice = config.lattice_arch(cfg.get("arch")) if cfg.get("arch") else None
        if lattice is None or "_x_" not in lattice:
            raise Unconvertible(f"arch={cfg.get('arch')!r} names no lattice map and the cfg "
                                f"records no lattice_modulus_map / lattice_argument_map")
        modulus, argument = lattice.split("_x_", 1)
        cfg.setdefault("lattice_modulus_map", modulus)
        cfg.setdefault("lattice_argument_map", argument)
    for key in present:
        cfg.pop(key)
    return True


def convert_value(value):
    """Every MIRA cfg inside `value`; True when anything changed."""
    changed = False
    if isinstance(value, list):
        for item in value:
            changed = convert_value(item) or changed
    elif isinstance(value, dict):
        if is_mira_cfg(value):
            changed = convert_cfg(value) or changed
        for key, item in value.items():
            if key not in SKIPPED_KEYS:
                changed = convert_value(item) or changed
    return changed


def still_carries(value):
    """Does any MIRA cfg inside `value` still record arch or pe?"""
    if isinstance(value, list):
        return any(still_carries(item) for item in value)
    if isinstance(value, dict):
        if is_mira_cfg(value) and any(key in value for key in DROPPED_KEYS):
            return True
        return any(still_carries(item) for key, item in value.items() if key not in SKIPPED_KEYS)
    return False


def load(path, as_json):
    if as_json:
        return json.loads(path.read_text())
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def convert_file(path, dry_run):
    """True when `path` holds a MIRA cfg with arch or pe (and, unless dry_run, now does not)."""
    as_json = path.suffix == ".json"
    value = load(path, as_json)
    if not convert_value(value):
        return False
    if dry_run:
        return True
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        if as_json:
            Path(temporary).write_text(json.dumps(value, indent=2) + "\n")
        else:
            import torch
            torch.save(value, temporary)
        if still_carries(load(Path(temporary), as_json)):
            raise RuntimeError("the written file does not read back converted")
        shutil.copystat(path, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return True


def mira_directories(roots):
    """Every directory under `roots` whose config.json is a MIRA model cfg."""
    found = set()
    for root in roots:
        sidecars = [root] if root.is_file() and root.name == "config.json" else (
            [root / "config.json"] if root.is_file() else root.rglob("config.json"))
        for sidecar in sidecars:
            try:
                if is_mira_cfg(json.loads(Path(sidecar).read_text())):
                    found.add(Path(sidecar).parent)
            except (OSError, ValueError):
                continue
    return found


def main(argv=None):
    from tqdm import tqdm

    arguments = list(argv if argv is not None else sys.argv[1:])
    dry_run = "--dry-run" in arguments
    paths = [Path(item).resolve() for item in arguments if item != "--dry-run"]
    defaults = [Path(config.CHECKPOINT_DIR), Path(config.MODEL_DIR)]
    roots = paths or [path for path in defaults if path.exists()]
    for root in roots:
        if not root.exists():
            print(f"[convert_ckpt] path does not exist: {root}")
            return 2
    directories = mira_directories(roots)
    files = {path for directory in directories for path in directory.iterdir()
             if path.is_file() and not path.name.endswith(".tmp")
             and (path.suffix in CHECKPOINT_SUFFIXES or path.name == "config.json")}
    print(f"[convert_ckpt] {len(directories)} MIRA run directory(ies); other directories are "
          f"not touched")
    changed, failures = 0, 0
    bar = tqdm(sorted(files), desc="converting checkpoints", unit="file", total=len(files))
    for path in bar:
        try:
            if convert_file(path, dry_run):
                changed += 1
                tqdm.write(f"[convert_ckpt] {'would convert' if dry_run else 'converted'} {path}")
        except Exception as error:                                      # noqa: BLE001
            failures += 1
            tqdm.write(f"[convert_ckpt] failed {path}: {error}")
        bar.set_postfix(converted=changed, failed=failures)
    print(f"[convert_ckpt] {'dry run: ' if dry_run else ''}scanned {len(files)} file(s); "
          f"{'to convert' if dry_run else 'converted'} {changed}; failed {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
