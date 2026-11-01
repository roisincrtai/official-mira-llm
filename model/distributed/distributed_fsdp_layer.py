"""
distributed_fsdp_layer.py -- distributed_parallelism: fsdp-layer. Spread one model's LAYERS across
the GPUs that are present, in one process.

    shard(model, devices, log)     put contiguous runs of blocks on each; returns the model
    is_sharded(model)              whether the above has been applied
    distribute(model, device, log) shard over every visible GPU; the model unchanged on one

WHAT IT DOES NOT DO. Only one device is busy at a time: device 1 waits while device 0 runs its
half, then device 0 waits while device 1 runs its. Two cards therefore give roughly the memory
of two and the speed of one. That is the trade, and it is worth making when a model or a
context window does not fit one card AT ALL -- which for mira-l is every window past 4,096.
When the model already fits, one card is faster and this should be turned off.

Filling the idle device requires splitting the batch into micro-batches and keeping several in
flight at once, so that device 1 works on micro-batch k while device 0 starts k+1. That is
pipeline scheduling (GPipe, 1F1B), and it is not done here.

fsdp-layer splits by layer and needs no collective. fsdp-channel (model/distributed/distributed_fsdp_channel.py)
splits individual weight matrices -- each rank holds part of every lattice and FFN projection
and the partial outputs are all-reduced -- which keeps every device busy on every layer at the
cost of a collective at each block.

WHY IT IS NOT DataParallel. torch.nn.DataParallel replicates the WHOLE model onto every
device, which needs the model to fit one card in the first place -- exactly the case this
exists for. It also is not DistributedDataParallel: that is one process per device, and this
is one process holding devices in sequence, which keeps every stage's resume, checkpointing
and logging working unchanged.
"""


from model.parallel import devices, dist_env, report_devices


def is_sharded(model):
    return bool(getattr(model, "_shard_devices", None))


def plan(n_blocks, n_devices):
    """Which device each block goes on: contiguous runs, the remainder on the EARLY devices.

    Contiguous because a boundary costs a copy, and `k` runs cost `k-1` copies however the
    blocks are grouped -- so the grouping that minimises copies is the one with the fewest
    runs, which is one run per device. The remainder goes early rather than late because the
    last device also carries the vocabulary projection, which at 50,259 columns outweighs a
    block several times over."""
    if n_devices <= 1 or n_blocks <= 0:
        return [0] * max(n_blocks, 0)
    base, extra = divmod(n_blocks, n_devices)
    out = []
    for i in range(n_devices):
        out += [i] * (base + (1 if i < extra else 0))
    return out[:n_blocks]


def shard(model, devs=None, log=print):
    """Place `model`'s blocks across `devs`. Returns the model, moved in place.

    THE EMBEDDING AND THE HEAD ARE ONE TENSOR, so they get one device -- the last, where the
    projection happens. `tok.weight is head.weight` is what weight tying means; putting the
    embedding on the first device and the head on the last would either break the tie, leaving
    two matrices that start equal and then quietly diverge, or fail outright. So the embedding
    lookup runs on the last device and its output is copied to the first block's, which is one
    (batch, tokens, d_model) copy -- the same size as every other boundary crossing, and the
    price of keeping the tie intact."""
    devs = list(devs if devs is not None else devices())
    if len(devs) < 2:
        return model
    where = plan(len(model.blocks), len(devs))
    for blk, i in zip(model.blocks, where):
        blk.to(devs[i])
    model.lnf.to(devs[-1])
    model.head.to(devs[-1])
    model.tok.to(devs[-1])
    model.tok.weight = model.head.weight              # the tie, re-established after the move
    model.drop.to(devs[-1])
    model.mask_embed.data = model.mask_embed.data.to(devs[-1])
    model._shard_devices = devs
    model._shard_where = where
    counts = [sum(1 for w in where if w == i) for i in range(len(devs))]
    log(f"[parallel] {len(model.blocks)} blocks across {len(devs)} devices: "
        + ", ".join(f"{d} {c} block(s)" for d, c in zip(devs, counts))
        + f"; embedding and head on {devs[-1]} (tied)")
    return model


def block_device(model, i):
    """The device block `i` lives on, or None when the model was never sharded."""
    if not is_sharded(model):
        return None
    return model._shard_devices[model._shard_where[i]]


def distribute(model, device=None, log=print):
    """distributed_parallelism: fsdp-layer. One process holds every visible GPU."""
    if dist_env()[2] > 1:
        raise RuntimeError("fsdp-layer requires one process controlling all visible GPUs")
    devs = devices()
    report_devices(log)
    if len(devs) < 2:
        return model.to(device) if device is not None else model
    return shard(model, devs, log=log)
