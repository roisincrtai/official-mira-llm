"""
parallel.py -- the shared entry for distributed training: distributed_parallelism is one of

    ddp            model/distributed/distributed_ddp.py: one process per GPU (torchrun); every rank
                   holds the whole model and reads its own share of the batch; gradients averaged
    fsdp-layer     model/distributed/distributed_fsdp_layer.py: one process holding every GPU;
                   contiguous runs of blocks on each
    fsdp-channel   model/distributed/distributed_fsdp_channel.py: one process per GPU (torchrun);
                   every rank reads the same batch; in each MIRA block rank r holds d_e/N lattice
                   channels and f/N FFN hidden units

and this module holds what every one of them calls:

    distribute(model, mode, device, log)  the model, distributed by `mode`
    devices(), report_devices()           the visible GPUs
    dist_env(), dist_init(), is_main()    torchrun's rank and world, the process group, rank 0
    data_env(mode)                        (rank, world) of the data split
    batch_layout(model, per_gpu)          the per-GPU, per-process and global batch
    broadcast_module(), broadcast_ckpt()  rank 0's model and checkpoint on every rank
    all_reduce_grads(params)              the per-step gradient reduction of the mode in use
    clip_grad_norm_(params, max_norm)     the clip on the norm of the full gradient
    all_reduce_mean(), barrier()          logging reduction, rank synchronisation
"""


def devices():
    """Every visible CUDA device, in order. Empty when there is no CUDA."""
    import torch
    from kernel.hardware import accelerator, resolve
    import os
    device = resolve(os.environ.get("GPU", "auto"))
    backend = accelerator(device)
    return [torch.device(device.type, i) for i in range(backend.device_count())] if backend else []


def report_devices(log=print):
    """Print EVERY GPU this process can see, one line each. Always, whatever the count.

    THE INDEX IS THE ONE THIS PROCESS USES. Under CUDA_VISIBLE_DEVICES=2 the card the driver
    calls 2 is cuda:0 here, so the number below is torch's, and the environment variable is
    printed beside it rather than left to be inferred."""
    import os
    try:
        import torch
    except Exception:                                              # noqa: BLE001
        log("[gpu] torch not importable")
        return []
    from kernel.hardware import resolve
    if resolve(os.environ.get("GPU", "auto")).type == "npu":
        names = [torch.npu.get_device_name(i) for i in range(torch.npu.device_count())]
        log(f"[gpu] {len(names)} Ascend device(s) detected")
        for i, name in enumerate(names):
            log(f"[gpu]   npu:{i}  {name}")
        return names
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not torch.cuda.is_available():
        log("[gpu] NO CUDA DEVICE VISIBLE -- training will run on CPU"
            + (f"   (CUDA_VISIBLE_DEVICES={vis!r})" if vis is not None else ""))
        return []
    n = torch.cuda.device_count()
    log(f"[gpu] {n} CUDA device(s) detected"
        + (f"   CUDA_VISIBLE_DEVICES={vis!r}" if vis is not None
           else "   CUDA_VISIBLE_DEVICES unset"))
    out = []
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        out.append(p.name)
        log(f"[gpu]   cuda:{i}  {p.name}  sm_{p.major}{p.minor}  "
            f"{p.total_memory / 1024 ** 3:.1f} GiB  {p.multi_processor_count} SMs")
    if n == 1:
        log("[gpu] 1 device: the model is not wrapped")
    return out


MODES = ("ddp", "fsdp-layer", "fsdp-channel")


def distribute(model, mode, device=None, log=print):
    """`model` distributed by `mode`, one of MODES."""
    mode = str(mode).lower()
    if mode == "ddp":
        from model.distributed import distributed_ddp as chosen
    elif mode == "fsdp-layer":
        from model.distributed import distributed_fsdp_layer as chosen
    elif mode == "fsdp-channel":
        from model.distributed import distributed_fsdp_channel as chosen
    else:
        raise ValueError(f"distributed_parallelism must be one of {', '.join(MODES)}, "
                         f"got {mode!r}")
    return chosen.distribute(model, device=device, log=log)


def channel_world():
    """The number of ranks fsdp-channel split the MIRA channels over; 1 when it did not."""
    from model.distributed import distributed_fsdp_channel
    return distributed_fsdp_channel.CHANNEL["world"]


def data_env(mode):
    """(rank, world) of the data split: under fsdp-channel every rank reads the same batch, so
    (0, 1); otherwise torchrun's rank and world size."""
    rank, _local, world = dist_env()
    return (0, 1) if str(mode).lower() == "fsdp-channel" else (rank, world)


def batch_layout(model, per_gpu):
    """Return ``(per_gpu, per_process, global_batch, n_gpu)`` for a trainer batch.

    ``BATCH`` is always sequences PER GPU. Under torchrun one process owns one GPU, so each
    process consumes ``per_gpu`` rows. Under the legacy one-process DataParallel path, that one
    process constructs ``per_gpu * n_gpu`` rows before the wrapper scatters them. The global
    optimiser step is ``per_gpu * n_gpu`` in both cases.
    """
    per_gpu = max(1, int(per_gpu))
    _, _, world = dist_env()
    if channel_world() > 1:
        return per_gpu, per_gpu, per_gpu, 1
    if world > 1:
        return per_gpu, per_gpu, per_gpu * world, world
    n_gpu = 1
    for module in model.modules():
        n_gpu = max(n_gpu, len(getattr(module, "device_ids", []) or []))
    return per_gpu, per_gpu * n_gpu, per_gpu * n_gpu, n_gpu


# =================================================================================================
# DISTRIBUTED DATA PARALLEL. One process per GPU, launched by torchrun.
#
# WHY, AND IT IS A MEASUREMENT RATHER THAN A PREFERENCE. nn.DataParallel gathers on GPU 0, so
# the master carries the optimiser state, the gathered activations AND the vocabulary
# projection. Measured on 2026-08-15 at batch 60 on 4 x 40 GiB:
#
#     worker  6,046 MiB = 0.17 GiB replica         + activations
#     master 11,964 MiB = 0.87 GiB optimiser state + activations + 5.08 GiB gather/loss
#
# so the master pays 1.89x a worker per unit of batch and saturates first. Every batch decision
# in this project has been set by that one card -- batch 200 projects to 40.4 GiB on the master
# while the workers sit at 38.5. Under DDP there is no master: each rank holds a replica, runs
# its own forward and backward, and the gradients are all-reduced. The asymmetry disappears and
# with it the constraint.
#
# THIS PART IS DELIBERATELY ONLY THE PLUMBING. It reads what torchrun sets and no-ops when
# nothing set it, so importing or calling it changes nothing about a single-process run. The
# trainer, the sharding contract in helpers/token_store.batch and rank-0 checkpointing are
# separate steps -- see user/Instructions.md, "The remaining work, in detail".
# =================================================================================================

def dist_env():
    """(rank, local_rank, world_size) from torchrun's environment, or (0, 0, 1).

    torchrun sets RANK, LOCAL_RANK and WORLD_SIZE in every worker. Their ABSENCE is the signal
    that this is an ordinary single-process run -- not an error, and not something to warn
    about, because every stage in this repository must still run that way."""
    import os
    try:
        return (int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]),
                int(os.environ["WORLD_SIZE"]))
    except (KeyError, ValueError):
        return 0, 0, 1


def dist_active():
    """Is a process group initialised? False on a single-process run, and False BEFORE
    dist_init -- so a caller can ask without needing to know the launch order."""
    try:
        import torch.distributed as d
        return d.is_available() and d.is_initialized()
    except Exception:                                              # noqa: BLE001
        return False


def is_main():
    """Rank 0, or any single-process run. What guards printing, checkpoint writes and figures:
    four ranks each writing the same checkpoint is four processes racing on one path."""
    return dist_env()[0] == 0


def _init_accepts(name):
    """Does THIS torch's init_process_group take this keyword? Cheap, and asked rather than
    assumed: `device_id` was added to it partway through torch's history, and a fixed argument
    list works on one release and raises TypeError on the next -- which would turn a missing
    keyword into a dead run on whichever cluster has the older build."""
    try:
        import inspect
        import torch.distributed as d
        return name in inspect.signature(d.init_process_group).parameters
    except Exception:                                              # noqa: BLE001
        return False


# "Process rank <N> started", said once per process. See dist_init.
_ANNOUNCED = False


def dist_init(log=print, device=None):
    """Join the process group and bind this rank to its GPU. Returns (rank, local, world).

    IDEMPOTENT AND SILENT WHEN NOT UNDER torchrun, so it can be called unconditionally from
    every stage. world_size 1 means there is nothing to join.

    THE DEVICE IS SET HERE AND NOWHERE ELSE. NCCL needs each rank pinned to a distinct device
    BEFORE the group is formed; ranks that all land on cuda:0 deadlock in init_process_group
    rather than failing, which is a hang with no message and no traceback.

    AND THE GROUP IS TOLD WHICH DEVICE THAT IS. A group formed without `device_id` does not know
    what card it belongs to, so every later device-bound collective has to infer one; barrier()
    infers it from the ambient context and says so --

        c10d_logger.py:83: UserWarning: barrier(): using the device under current context. You
        can specify `device_id` in `init_process_group` to mute this warning.

    -- which is not a cosmetic complaint. The context is whatever the calling thread last set,
    and dist_close() barriers at interpreter exit, long after the training loop that set it; a
    guess that lands on the wrong card puts the barrier on a device this rank does not own.
    Naming the device is the fix, not the silencer: it also lets NCCL bind its communicator
    eagerly instead of on the first collective.

    EVERY RANK SAYS IT STARTED, AND THAT IS A DELIBERATE EXCEPTION TO "print only in the main
    process":

    Every other routine line in this pipeline is gated on rank 0 (helpers.common._say,
    driver.main_process) because four copies of one line interleave on one terminal and stop
    being readable. This one is the opposite case: it is USEFUL exactly once per process, and
    what it tells you is which ranks exist. A launch that hangs -- the classic one is two ranks
    landing on the same card, which deadlocks INSIDE init_process_group with no message and no
    traceback -- prints three lines instead of four, and the missing number is the answer.

    SO IT IS PRINTED BEFORE THE GROUP IS FORMED, not after: a line after init_process_group
    would be a line the hang eats. `print` and not `log`, because a stage's `log` is the rank-0
    gate and this line is the one thing that must escape it.

    ONLY UNDER DDP. world <= 1 is a single process -- with or without torchrun -- where there is
    no group, nothing to hang in, and no other rank to be missing; there is nothing for a rank
    number to distinguish it from. Once per process, because dist_init is idempotent and every
    stage calls it."""
    rank, local, world = dist_env()
    if world <= 1:
        return rank, local, world
    global _ANNOUNCED
    if not _ANNOUNCED:
        _ANNOUNCED = True
        print(f"Process rank {rank} started", flush=True)
    import torch
    import torch.distributed as d
    from kernel.hardware import resolve, accelerator
    import os
    selected = resolve(os.environ.get("GPU", "auto")) if device is None else torch.device(device)
    backend = {"cuda": "nccl", "npu": "hccl"}.get(selected.type, "gloo")
    if not d.is_initialized():
        card = accelerator(selected)
        if card is not None:
            card.set_device(local)
        # nccl for CUDA, gloo otherwise -- the CPU path exists so a laptop can run the same
        # code, not because it is fast.
        kw = {}
        if selected.type == "cuda" and _init_accepts("device_id"):
            # ONLY ON THE NCCL PATH. A gloo group is not bound to a card at all, and handing it
            # a CUDA device names a device its collectives will never run on.
            kw["device_id"] = torch.device("cuda", local)
        d.init_process_group(backend=backend, **kw)
        if rank == 0:
            log(f"[parallel] torchrun: {world} rank(s), backend "
                f"{backend}, one process per GPU")
    return rank, local, world


def dist_close():
    """Leave the process group. Worth doing explicitly: a rank that exits without it can leave
    the others blocked in a collective until the launcher times them out."""
    try:
        import torch.distributed as d
        if d.is_available() and d.is_initialized():
            d.destroy_process_group()
    except Exception:                                              # noqa: BLE001
        pass


def all_reduce_mean(x):
    """The mean of a scalar across ranks, for LOGGING. Returns x unchanged off DDP, and under
    fsdp-channel, where every rank computed the same value from the same batch.

    THE LOSS NEEDS THIS AND SO DO THE GRADIENTS, by a different route: all_reduce_grads below
    averages those. The loss value each rank computes is over its OWN shard -- so a curve
    logged without this describes only rank 0's per-GPU batch, rather than the mean over the
    global batch, and every history record, figure and single-GPU comparison then measures a
    different quantity."""
    if not dist_active() or channel_world() > 1:
        return x
    import torch
    import torch.distributed as d
    t = x if torch.is_tensor(x) else torch.tensor(float(x), device=_coll_device())
    t = t.detach().clone().float()
    d.all_reduce(t, op=d.ReduceOp.SUM)
    t /= d.get_world_size()
    return t if torch.is_tensor(x) else float(t)


# The reduction buffer, in ELEMENTS, not a knob: the flat fp32 copy a bucket needs is 4 bytes
# each, so 16 M elements is a 64 MiB staging buffer whatever the model's size. Small enough to
# be irrelevant beside the activations, large enough that a 20-block model is a handful of
# messages rather than one per tensor.
_REDUCE_BUCKET = 16 * 1024 * 1024


def broadcast_module(model, src=0):
    """Make every rank's copy of `model` bit-identical to `src`'s. Returns how many tensors were
    sent; a no-op returning 0 off DDP.

    THIS IS NOT OPTIONAL AND IT IS NOT DEFENSIVE. Nothing in this pipeline calls
    torch.manual_seed, so torch's global generator is seeded from the operating system and FOUR
    PROCESSES BUILD FOUR DIFFERENT MODELS. DistributedDataParallel's constructor hid that by
    broadcasting rank 0's parameters and buffers; without it, averaging gradients taken at four
    different points in weight space is not data parallelism, and the checkpoint rank 0 writes
    is a model that was steered by three others it never agreed with. Nothing about that is
    visible in a loss curve.

    PARAMETERS AND BUFFERS BOTH, which is what DDP synchronised at construction. ONCE, HERE,
    AND NEVER AGAIN: every buffer in this model is non-persistent and either constant
    (pe.inv_freq, pe.pe) or a diagnostic (mira._pi_v), so none of them is training state and
    none has to be kept in step during the run. A per-forward buffer sync would be a collective
    inside every forward -- and a forward that one rank makes and the others do not, which is
    what a rank-0 generation preview is, would then hang instead of printing.

    ONE BROADCAST PER TENSOR, IN parameters()-THEN-buffers() ORDER, which is identical on every
    rank because the architecture is. parameters() de-duplicates, so the tied embedding/head is
    sent once."""
    if not dist_active():
        return 0
    import torch
    import torch.distributed as d
    n = 0
    with torch.no_grad():
        for t in list(model.parameters()) + list(model.buffers()):
            d.broadcast(t.data, src=src)
            n += 1
    dev = _coll_device()
    cpu_rng = torch.get_rng_state().to(dev)
    d.broadcast(cpu_rng, src=src)
    torch.set_rng_state(cpu_rng.cpu())
    if torch.cuda.is_available():
        cuda_rng = torch.cuda.get_rng_state().to(dev)
        d.broadcast(cuda_rng, src=src)
        torch.cuda.set_rng_state(cuda_rng.cpu())
    return n


class _TensorSlot:
    """A tensor's SHAPE AND DTYPE, standing in the tensor's place inside a checkpoint skeleton.

    Small, picklable, and enough for a receiving rank to allocate the tensor it is about to be
    sent. Deliberately NOT the tensor's device: every tensor in a checkpoint is on the CPU (see
    helpers/utils.fp32_state -- "the file must not know what hardware wrote it") and every one
    of them is put back on the CPU here, so a file broadcast under four ranks is the file one
    rank would have read."""

    def __init__(self, shape, dtype):
        self.shape, self.dtype = shape, dtype


def _coll_device():
    """Where a staging tensor for a collective must live: this rank's card under nccl, the CPU
    under gloo. NCCL cannot broadcast a CPU tensor; gloo on a machine with no CUDA has nowhere
    else to put one. The card is the one dist_init pinned, so this asks nothing of the
    environment that has not already been decided."""
    import torch
    import torch.distributed as d
    try:
        if str(d.get_backend()) == "hccl":
            return torch.device("npu", torch.npu.current_device())
        nccl = str(d.get_backend()) == "nccl"
    except Exception:                                              # noqa: BLE001
        nccl = torch.cuda.is_available()
    if nccl and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _skeleton(obj, out):
    """`obj` with every tensor replaced by a _TensorSlot, the tensors appended to `out` in the
    order they were met.

    THE CONTAINERS ARE REBUILT, NOT MUTATED, so the caller's checkpoint is untouched and can
    still be the thing rank 0 returns. Dicts, lists and tuples are the only containers a
    checkpoint written by save_ckpt contains -- a state_dict, the optimiser's
    {"state": ..., "param_groups": [...]}, the list of generator states, and the scalars."""
    import torch
    if torch.is_tensor(obj):
        out.append(obj)
        return _TensorSlot(tuple(obj.shape), obj.dtype)
    if isinstance(obj, dict):
        return {k: _skeleton(v, out) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_skeleton(v, out) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_skeleton(v, out) for v in obj)
    return obj


def _fill(sk, take):
    """The inverse of _skeleton: every _TensorSlot replaced by `take(slot)`.

    THE SAME TRAVERSAL, WHICH IS THE WHOLE POINT. Rank 0 and every other rank walk the same
    skeleton with the same recursion, so `take` is called the same number of times in the same
    order on all of them -- and `take` is where the broadcast is. The symmetry of the collective
    is therefore a property of this function rather than something the call sites have to
    maintain."""
    if isinstance(sk, _TensorSlot):
        return take(sk)
    if isinstance(sk, dict):
        return {k: _fill(v, take) for k, v in sk.items()}
    if isinstance(sk, list):
        return [_fill(v, take) for v in sk]
    if isinstance(sk, tuple):
        return tuple(_fill(v, take) for v in sk)
    return sk


def broadcast_ckpt(ck, src=0):
    """`src`'s loaded checkpoint, on every rank. Returns `ck` unchanged off DDP.

    CALLED ON EVERY RANK OR ON NONE. It is three families of collective -- the skeleton's
    length, the skeleton, and one broadcast per tensor in it -- so a rank that skips it leaves
    the others waiting in a broadcast with no message coming, which is a hang and not a crash.
    The one caller is helpers/utils.load_ckpt, which is guarded by dist_active() so that
    mirachat.py, the evals and any other single-process reader never reach a collective at all.

    `ck` IS IGNORED ON EVERY RANK BUT `src`, and is None there. Only rank 0 touched the disk, so
    only rank 0 knows whether there was a file to read: a checkpoint that does not exist is a
    skeleton of None, which costs the two small messages and no tensor ones, and every rank
    returns None together.

    WHAT COMES BACK IS ON THE CPU AND IS THE SAME SHAPE OF OBJECT EVERYWHERE. Rank 0 rebuilds
    its own dictionary through the same _fill as everyone else, holding on to the tensors it
    already has rather than copying them, so no rank gets a structure the others do not."""
    if not dist_active():
        return ck
    import pickle
    import torch
    import torch.distributed as d
    dev = _coll_device()
    mine = (d.get_rank() == src)
    tensors = []
    sk = _skeleton(ck, tensors) if mine else None
    blob = pickle.dumps(sk, protocol=pickle.HIGHEST_PROTOCOL) if mine else b""
    # THE LENGTH FIRST, IN ITS OWN MESSAGE. A receiving rank has to allocate the buffer before
    # it can be filled, and it cannot know the size any other way.
    n = torch.tensor([len(blob)], dtype=torch.int64, device=dev)
    d.broadcast(n, src=src)
    size = int(n.item())
    # `raw` IS NAMED SO THAT IT OUTLIVES THE BROADCAST. torch.frombuffer does not copy -- the
    # tensor is a view of these bytes -- and a bytearray built inline inside the call is a
    # temporary. Under gloo, where dev is the CPU and `.to` is a no-op, that view IS the tensor
    # being sent.
    raw = bytearray(blob) if mine else None
    buf = (torch.frombuffer(raw, dtype=torch.uint8).to(dev) if mine
           else torch.empty(size, dtype=torch.uint8, device=dev))
    d.broadcast(buf, src=src)
    if not mine:
        sk = pickle.loads(buf.cpu().numpy().tobytes())
    del buf
    it = iter(tensors)

    def take(slot):
        # ONE TENSOR THROUGH THE STAGING DEVICE AT A TIME. The peak is the largest tensor in the
        # checkpoint -- the vocabulary projection -- and not the checkpoint.
        t = next(it) if mine else None
        s = (t.detach().contiguous().to(dev) if t is not None
             else torch.empty(slot.shape, dtype=slot.dtype, device=dev))
        d.broadcast(s, src=src)
        return t if mine else s.to("cpu")

    return _fill(sk, take)


def all_reduce_grads(params):
    """The gradient reduction of the mode in use, once per optimiser step, after the last
    micro-batch's backward and before the clip. Returns how many tensors were reduced; 0 off
    torchrun. ddp averages every gradient (distributed_ddp.average_grads); fsdp-channel keeps
    each rank's slice and reduces the whole parts (distributed_fsdp_channel.reduce_grads)."""
    if not dist_active():
        return 0
    if channel_world() > 1:
        from model.distributed import distributed_fsdp_channel
        return distributed_fsdp_channel.reduce_grads(params)
    from model.distributed import distributed_ddp
    return distributed_ddp.average_grads(params)


def clip_grad_norm_(parameters, max_norm):
    """torch.nn.utils.clip_grad_norm_ on the norm of the FULL gradient; under fsdp-channel the
    split parts are summed over the ranks (distributed_fsdp_channel.clip_grad_norm_)."""
    import torch
    parameters = list(parameters)
    if channel_world() > 1 and dist_active():
        from model.distributed import distributed_fsdp_channel
        return distributed_fsdp_channel.clip_grad_norm_(parameters, max_norm)
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def barrier(log=None):
    """Wait for every rank. No-op off DDP.

    NEEDED WHEREVER ONE RANK WRITES A FILE THE OTHERS WILL READ. Rank 0 saves the checkpoint;
    without a barrier a faster rank reaches the next resume point and reads a file that is
    still being written -- and _atomic_torch_save's rename means it reads the PREVIOUS
    checkpoint rather than a truncated one, so the failure is a silent step backwards rather
    than an exception."""
    try:
        import torch.distributed as d
        if d.is_available() and d.is_initialized():
            d.barrier()
    except Exception:                                              # noqa: BLE001
        pass
