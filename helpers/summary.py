"""helpers/summary.py -- one summary.json per run, never one for the sweep.

A run owns what it produced. Its checkpoint, its history, its dynamics and its held-out score all
sit in its own directory, and its summary record belongs beside them. A single file at the root of
the sweep made the SWEEP the unit of storage instead of the RUN, with three consequences:

Per run, a rewrite touches one record, an interrupt costs one record, and a removed directory
removes exactly its own result.

READING MERGES, WRITING IS LOCAL. `collect` walks the unit directories; `write` touches only the
unit it was given.
"""

import json
import os

NAME = "summary.json"


def path(root, unit):
    """<root>/<unit>/summary.json -- beside that run's checkpoint."""
    return os.path.join(root, unit, NAME)


def write(root, unit, record):
    """One run's record, through a temporary so an interrupt leaves the old file or the new one."""
    target = path(root, unit)
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    tmp = f"{target}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)


def collect(root, units=None):
    """{unit: record} merged from the unit directories, skipping any that is absent or unreadable.

    `units` restricts the walk to a known set; without it every directory under `root` is tried,
    which is what a redraw wants when it has no plan in hand.
    """
    out = {}
    if not os.path.isdir(root):
        return out
    for unit in sorted(units if units is not None else os.listdir(root)):
        target = path(root, unit)
        if not os.path.isfile(target):
            continue
        try:
            with open(target, "r", encoding="utf-8") as handle:
                out[unit] = json.load(handle)
        except (OSError, ValueError, TypeError):
            continue
    return out


def split_legacy(root, units, legacy=None, log=None):
    """Split a sweep-level summary.json into per-unit files, then retire it.

    NOTHING IS OVERWRITTEN AND NOTHING IS DELETED. A unit that already has its own file is left
    alone, and the shared file is renamed rather than removed once EVERY record has a home."""
    legacy = legacy or {}
    shared = os.path.join(root, NAME)
    if not os.path.isfile(shared):
        return 0
    try:
        with open(shared, "r", encoding="utf-8") as handle:
            records = json.load(handle)
    except (OSError, ValueError, TypeError):
        return 0
    if not isinstance(records, list):
        return 0
    known = set(units)
    written, left = 0, 0
    for record in records:
        unit = record.get("variant") or record.get("family")
        unit = legacy.get(unit, unit)
        if unit not in known or not os.path.isdir(os.path.join(root, unit)):
            left += 1
            continue
        if os.path.exists(path(root, unit)):
            continue                                    # already placed; not left behind
        record = dict(record)
        if "variant" in record:
            record["variant"] = unit
        elif "family" in record:
            record["family"] = unit
        write(root, unit, record)
        written += 1
    if written and not left:
        os.replace(shared, f"{shared}.retired")
        if log:
            log(f"split {written} summary record(s) into per-run files; "
                f"the shared file is now {NAME}.retired")
    elif written and log:
        log(f"split {written} summary record(s) into per-run files; {left} record(s) belong to "
            f"units not in this run, so {NAME} is kept until a run can place them")
    return written
