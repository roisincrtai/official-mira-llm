"""
distributed_fsdp_channel.py -- distributed_parallelism: fsdp-channel.

    fsdp-channel   one process per GPU (torchrun); every rank reads the same batch; in each MIRA
                   block rank r holds d_e/N lattice channels and f/N FFN hidden units

    distribute(model, device, log)       split every MIRA block over the ranks (rank 0's model,
                                         its split parts scattered and the rest broadcast)
    reduce_grads(params)                 the per-step gradient reduction
    clip_grad_norm_(parameters, max)     the norm of the full gradient
    channel_fp32_state, channel_optimizer_state, channel_local_optimizer_state
                                         the checkpoint in the full layout, written and read
"""


from model.parallel import (_REDUCE_BUCKET, _coll_device, dist_active, dist_env, dist_init,
                            is_main, report_devices)


# =================================================================================================
# fsdp-channel. One process per GPU, launched by torchrun. Every rank reads the same batch.
#
# In each MIRA block (model/mira_block.MiraBlock) rank r of N holds:
#
#     input_projection   rows of its d_e/N signal channels, and the d_m write rows whole
#     to_gate            rows of its d_e/N channels
#     to_control         whole
#     coefficient_proj.  alpha, beta, rho rows of its d_e/N channels, and the (K+1) d_m read rows
#                        whole
#     out_proj           columns of its d_e/N channels: a partial (B, T, d)
#     ffn.up             rows of its f/N hidden units
#     ffn.down           columns of its f/N hidden units: a partial (B, T, d); the bias is added
#                        once, after the sum
#
# and the residual stream, the two RMSNorms, the embedding, the head and the final norm whole.
#
# INTER-GPU COMMUNICATION, PER BLOCK PER STEP: one sum of the mixer's partial outputs and one of
# the FFN's in the forward, and one all-reduce of the mixer's and one of the FFN's input gradient
# in the backward. A sum is accumulated in fp32; an output narrower than fp32 is reduce-scattered
# in fp32 and its summed shards all-gathered in its own dtype. With gradient checkpointing the
# block recomputes only the work between those sums, which are kept, so no forward sum is sent
# twice. The scan, the write, read and control projections and the gate need nothing.
#
# PER STEP: the gradients of the whole parts that see only the local channels (to_control, the
# write and read rows) are summed over the ranks; every other whole parameter's gradient is
# rank 0's, broadcast, so the copies stay bit-identical; the clip sums one scalar.
#
# AT BUILD rank 0's model is the model: its split parts are scattered, each rank receiving only
# its own, and the rest broadcast. A checkpoint gathers the split parts on rank 0 alone.
# =================================================================================================

CHANNEL = {"rank": 0, "world": 1}


class ChannelSpec:
    """How the local tensor of a split parameter maps onto the full tensor, along `dim`.

    `segments` are (kind, length) and cover the full length in order. A "split" segment of length
    n is cut into N contiguous parts of n/N and rank r holds part r; a "whole" segment is held
    whole by every rank. The local tensor is the held parts concatenated in segment order."""

    def __init__(self, dim, segments):
        self.dim = int(dim)
        self.segments = tuple((str(kind), int(n)) for kind, n in segments)

    def ranges(self, world, kind):
        """(offset, length) in the local tensor of every segment of `kind`."""
        out, off = [], 0
        for k, n in self.segments:
            length = n // world if k == "split" else n
            if k == kind:
                out.append((off, length))
            off += length
        return out

    def local(self, full, rank, world):
        import torch
        pieces, lo = [], 0
        for kind, n in self.segments:
            if kind == "split":
                part = n // world
                pieces.append(full.narrow(self.dim, lo + rank * part, part))
            else:
                pieces.append(full.narrow(self.dim, lo, n))
            lo += n
        return torch.cat(pieces, dim=self.dim)

    def full(self, parts):
        import torch
        world = len(parts)
        pieces, off = [], 0
        for kind, n in self.segments:
            if kind == "split":
                part = n // world
                pieces += [p.narrow(self.dim, off, part) for p in parts]
                off += part
            else:
                pieces.append(parts[0].narrow(self.dim, off, n))
                off += n
        return torch.cat(pieces, dim=self.dim)


def _all_gather_flat(out, shard):
    import torch.distributed as d
    gather = getattr(d, "all_gather_single", None) or d.all_gather_into_tensor
    gather(out, shard)


def _sum_over_ranks(t):
    """The sum of `t` over the ranks, accumulated in fp32 (fp64 for fp64) and returned in t's
    dtype. A tensor narrower than fp32 is reduce-scattered in fp32 and the summed shards are
    all-gathered in its own dtype; any other is all-reduced."""
    import torch
    import torch.distributed as d
    world = CHANNEL["world"]
    wide = torch.float64 if t.dtype == torch.float64 else torch.float32
    flat = t.detach().reshape(-1).to(wide)
    if t.dtype == wide:
        flat = flat.clone()
        d.all_reduce(flat)
        return flat.view(t.shape)
    count = flat.numel()
    pad = (-count) % world
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    shard = flat.new_empty(flat.numel() // world)
    d.reduce_scatter_tensor(shard, flat)
    out = torch.empty(flat.numel(), dtype=t.dtype, device=t.device)
    _all_gather_flat(out, shard.to(t.dtype))
    return out[:count].view(t.shape)


def _to_channels(x):
    import torch

    class ToChannels(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.view_as(value)

        @staticmethod
        def backward(ctx, grad):
            return _sum_over_ranks(grad)

    return ToChannels.apply(x)


def _from_channels(x):
    import torch

    class FromChannels(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return _sum_over_ranks(value)

        @staticmethod
        def backward(ctx, grad):
            return grad

    return FromChannels.apply(x)


def _mixer_sum(block, partial):
    """The mixer's output: the partial outputs summed. Its profile's output RMS is the summed
    output's."""
    from model import mira_profiling
    total = _from_channels(partial)
    profile = getattr(block.mixer, "_profile_ext", None)
    if mira_profiling.enabled() and isinstance(profile, dict) and "mixer_out_rms" in profile:
        profile["mixer_out_rms"] = mira_profiling._rms(total)
    return total


def _ffn_sum(block, partial):
    """The FFN's output: the partial outputs summed, then ffn.down's bias."""
    total = _from_channels(partial)
    bias = getattr(getattr(block.ffn, "down", None), "bias", None)
    return total if bias is None else total + bias.to(total.dtype)


def _first_call_profiled(fn):
    """`fn`, with the MIRA profile switched off from its second call on: the recomputation in
    the backward does not measure the step again."""
    from model import mira_profiling
    calls = [0]

    def run(*args):
        calls[0] += 1
        if calls[0] == 1:
            return fn(*args)
        was = mira_profiling._PROFILE["on"]
        mira_profiling._PROFILE["on"] = False
        try:
            return fn(*args)
        finally:
            mira_profiling._PROFILE["on"] = was

    return run


def _block_forward(self, x, attn_mask=None, cache=None, layer=None):
    x = x + _mixer_sum(self, self.mixer(self.rn1(x), attn_mask, cache, layer))
    return x + _ffn_sum(self, self.ffn(self.rn2(x)))


def _block_checkpointed(self, x, mask, cache, layer):
    """The block under gradient checkpointing: each partial output is recomputed in the
    backward, and each sum is kept."""
    from torch.utils.checkpoint import checkpoint
    mixer_part = _first_call_profiled(
        lambda h, m: self.mixer(self.rn1(h), m, cache, layer))
    partial = checkpoint(mixer_part, x, mask, use_reentrant=False, preserve_rng_state=True)
    x = x + _mixer_sum(self, partial)
    partial = checkpoint(lambda h: self.ffn(self.rn2(h)), x,
                         use_reentrant=False, preserve_rng_state=True)
    return x + _ffn_sum(self, partial)


_CHANNEL_CLASSES = {}


def _channel_class(base):
    """`base` made a part of an fsdp-channel block; one class per base, made once."""
    if base in _CHANNEL_CLASSES:
        return _CHANNEL_CLASSES[base]
    import torch.nn.functional as F
    from model import mira_block, mira_mixer
    if issubclass(base, mira_block.MiraBlock):
        body = {"forward": _block_forward, "channel_checkpointed": _block_checkpointed}
    elif issubclass(base, mira_mixer.MiraMixer):
        def forward(self, normalized_x, *args, **kwargs):
            return base.forward(self, _to_channels(normalized_x), *args, **kwargs)

        def state_values(self):
            return self.order * self.full_inner * self.memory_dimension

        body = {"forward": forward, "state_values": state_values}
    elif issubclass(base, (mira_block.FeedForward, mira_block.MoEFeedForward)):
        def forward(self, x):
            return base.forward(self, _to_channels(x))

        body = {"forward": forward}
    else:
        def forward(self, x):
            return F.linear(x, self.weight)

        body = {"forward": forward}
    cls = type("Channel" + base.__name__, (base,), body)
    cls.__module__ = __name__
    globals()[cls.__name__] = cls
    _CHANNEL_CLASSES[base] = cls
    return cls


def _split_parameter(module, name, spec, local):
    import torch.nn as nn
    old = getattr(module, name)
    if old is None:
        return
    new = nn.Parameter(local.detach().clone(), requires_grad=old.requires_grad)
    new.__dict__.update(old.__dict__)
    new._channel_spec = spec
    new._channel_full = tuple(old.shape)
    setattr(module, name, new)


def _slice_on_load(module, state_dict, prefix, local_metadata, strict, missing_keys,
                   unexpected_keys, error_msgs):
    """A full-shaped tensor in `state_dict` for a split parameter is cut to this rank's part."""
    rank, world = CHANNEL["rank"], CHANNEL["world"]
    for name, parameter in module.named_parameters():
        spec = getattr(parameter, "_channel_spec", None)
        key = prefix + name
        if spec is None or key not in state_dict:
            continue
        value = state_dict[key]
        if tuple(value.shape) == parameter._channel_full:
            state_dict[key] = spec.local(value, rank, world)


def _block_plan(block, world):
    """(module, parameter name, spec) of every split or partly split parameter of `block`."""
    from model import mira_block
    mixer, ffn = block.mixer, block.ffn
    channels, order = mixer.inner, mixer.order
    moe = isinstance(ffn, mira_block.MoEFeedForward)
    memory = mixer.memory_dimension
    hidden = ffn.expert_width if moe else ffn.up.out_features
    if channels % world or hidden % world:
        raise ValueError(f"fsdp-channel needs d_e ({channels}) and the FFN width ({hidden}) to "
                         f"be divisible by the {world} ranks")
    signal = ChannelSpec(0, (("split", channels), ("whole", memory)))
    gate = ChannelSpec(0, (("split", channels),))
    control = ChannelSpec(0, (("whole", mixer.coefficient_bottleneck),))
    coefficient = ChannelSpec(0, (("split", channels * order), ("split", channels * order),
                                  ("split", channels * (order + 1)),
                                  ("whole", (order + 1) * memory)))
    out = ChannelSpec(1, (("split", channels),))
    up = ChannelSpec(0, (("split", hidden),))
    down = ChannelSpec(1, (("split", hidden),))
    ffn_parts = (((ffn, ChannelSpec(1, (("split", hidden),)),
                   ("expert_up", "expert_up_bias", "shared_up", "shared_up_bias")),
                  (ffn, ChannelSpec(2, (("split", hidden),)), ("expert_down", "shared_down")),
                  (ffn, ChannelSpec(0, (("whole", ffn.experts),)), ("router",)))
                 if moe else
                 ((ffn.up, up, ("weight", "bias")),
                  (ffn.down, down, ("weight",))))
    plan = []
    for module, spec, names in ((mixer.input_projection, signal, ("weight", "bias")),
                                (mixer.to_gate, gate, ("weight", "bias")),
                                (mixer.to_control, control, ("weight", "bias")),
                                (mixer.coefficient_projection, coefficient, ("weight", "bias")),
                                (mixer.out_proj, out, ("weight",))) + ffn_parts:
        for name in names:
            if getattr(module, name) is not None:
                plan.append((module, name, spec))
    return plan


def _split_block(block, rank, world, received):
    from model import mira_block
    mixer, ffn = block.mixer, block.ffn
    channels, order = mixer.inner, mixer.order
    moe = isinstance(ffn, mira_block.MoEFeedForward)
    memory = mixer.memory_dimension
    hidden = ffn.expert_width if moe else ffn.up.out_features
    part, hidden_part = channels // world, hidden // world
    for module, name, spec in _block_plan(block, world):
        _split_parameter(module, name, spec, received[id(getattr(module, name))])
    mixer.input_projection.out_features = part + memory
    mixer.to_gate.out_features = part
    mixer.coefficient_projection.out_features = 3 * part * order + part + (order + 1) * memory
    mixer.out_proj.in_features = part
    if moe:
        ffn.expert_width = hidden_part
        ffn.down_bias_here = rank == 0
    else:
        ffn.up.out_features = hidden_part
        ffn.down.in_features = hidden_part
    mixer.full_inner = channels
    mixer.inner = part
    mixer.signal_span = (0, part)
    mixer.write_span = (part, part + memory)
    mixer.alpha_span = (0, part * order)
    mixer.beta_span = (part * order, 2 * part * order)
    mixer.rho_span = (2 * part * order, 2 * part * order + part * (order + 1))
    mixer.read_span = (mixer.rho_span[1], mixer.rho_span[1] + (order + 1) * memory)
    mixer._channel = (rank, world)
    if not moe:
        ffn.down.__class__ = _channel_class(type(ffn.down))
    mixer.__class__ = _channel_class(type(mixer))
    ffn.__class__ = _channel_class(type(ffn))
    block.__class__ = _channel_class(type(block))
    mixer._register_load_state_dict_pre_hook(_slice_on_load, with_module=True)
    ffn._register_load_state_dict_pre_hook(_slice_on_load, with_module=True)


def _scatter_or_broadcast(model, plan, rank, world, src=0):
    """Rank `src`'s model on every rank: each split parameter scattered (rank r receives part r),
    every other parameter and buffer broadcast, one tensor at a time through the collective
    device. Returns {id(parameter): this rank's part} for the split ones."""
    import torch
    import torch.distributed as d
    dev = _coll_device()
    received = {}
    with torch.no_grad():
        for parameter in model.parameters():
            spec = plan.get(id(parameter))
            if spec is None:
                staged = parameter.detach().to(dev)
                d.broadcast(staged, src=src)
                parameter.copy_(staged.to(parameter.device))
                continue
            mine = torch.empty_like(spec.local(parameter.detach(), rank, world), device=dev)
            parts = ([spec.local(parameter.detach(), r, world).contiguous().to(dev)
                      for r in range(world)] if rank == src else None)
            d.scatter(mine, parts, src=src)
            received[id(parameter)] = mine.to(parameter.device)
        for buffer in model.buffers():
            staged = buffer.detach().to(dev)
            d.broadcast(staged, src=src)
            buffer.copy_(staged.to(buffer.device))
    cpu_rng = torch.get_rng_state().to(dev)
    d.broadcast(cpu_rng, src=src)
    torch.set_rng_state(cpu_rng.cpu())
    if torch.cuda.is_available():
        cuda_rng = torch.cuda.get_rng_state().to(dev)
        d.broadcast(cuda_rng, src=src)
        torch.cuda.set_rng_state(cuda_rng.cpu())
    return received


def split_channels(model, rank, world, log=print):
    """Split every MIRA block of `model` over `world` ranks, in place; returns how many blocks
    were split. Every rank must call it with the same architecture; rank 0's weights are used."""
    import torch.nn as nn
    from model import mira_block
    for module in model.modules():
        if isinstance(module, nn.Dropout) and float(module.p) > 0.0:
            raise ValueError(f"fsdp-channel needs dropout 0.0; a dropout has p={module.p}")
    blocks = [m for m in model.modules() if isinstance(m, mira_block.MiraBlock)]
    plan = {}
    for block in blocks:
        for module, name, spec in _block_plan(block, world):
            plan[id(getattr(module, name))] = spec
    received = _scatter_or_broadcast(model, plan, rank, world)
    CHANNEL["rank"], CHANNEL["world"] = int(rank), int(world)
    for block in blocks:
        _split_block(block, rank, world, received)
    return len(blocks)


def distribute(model, device=None, log=print):
    """distribute(model, "fsdp-channel"). Under torchrun the MIRA blocks are split over the
    ranks and the model is moved to this rank's card; a single process has nothing to split."""
    rank, local, world = dist_env()
    if world <= 1:
        report_devices(log)
        return model.to(device) if device is not None else model
    dist_init(log)
    import torch
    import os
    from kernel.hardware import resolve
    selected = resolve(os.environ.get("GPU", "auto"))
    target = (torch.device(selected.type, local) if selected.type in ("cuda", "npu")
              else selected)
    model = model.to("cpu")
    split = split_channels(model, rank, world, log)
    model = model.to(target)
    if rank == 0:
        report_devices(log)
        if split:
            first = next(m for m in model.modules() if hasattr(m, "full_inner"))
            ffn = first_ffn(model)
            units = ffn.expert_width if hasattr(ffn, "expert_width") else ffn.up.out_features
            log(f"[parallel] fsdp-channel over {world} rank(s): {split} MIRA block(s), each rank "
                f"{first.inner:,} of {first.full_inner:,} lattice channels and "
                f"{units:,} of {units * world:,} FFN units; every rank reads "
                f"the same batch")
        else:
            log(f"[parallel] fsdp-channel over {world} rank(s): {type(model).__name__} has no "
                f"MIRA block; every rank runs the whole model on the same batch")
    return model


def first_ffn(model):
    from model import mira_block
    return next(m.ffn for m in model.modules() if isinstance(m, mira_block.MiraBlock))


def full_numel(parameter):
    """The element count of the full parameter, whether or not fsdp-channel split it."""
    shape = getattr(parameter, "_channel_full", None)
    if shape is None:
        return int(parameter.numel())
    count = 1
    for n in shape:
        count *= int(n)
    return count


def _buckets(tensors):
    buckets, cur, cur_n = [], [], 0
    for t in tensors:
        cur.append(t)
        cur_n += t.numel()
        if cur_n >= _REDUCE_BUCKET:
            buckets.append(cur)
            cur, cur_n = [], 0
    if cur:
        buckets.append(cur)
    return buckets


def _flat_collective(tensors, collective):
    """`collective` applied in place to `tensors`, through fp32 buckets."""
    import torch
    with torch.no_grad():
        for bucket in _buckets(tensors):
            flat = torch.cat([t.detach().reshape(-1).float() for t in bucket])
            collective(flat)
            off = 0
            for t in bucket:
                n = t.numel()
                t.copy_(flat[off:off + n].view(t.shape))
                off += n


def reduce_grads(params):
    """all_reduce_grads under fsdp-channel. Every rank runs the same graph, so the same
    parameters have a gradient. The split rows and columns of a parameter keep their own
    gradient; its whole segments, which see only the local channels, are summed over the ranks;
    a parameter with no split takes rank 0's gradient, which every rank computed from the same
    inputs. Returns how many tensors were reduced or sent."""
    import torch.distributed as d
    world = CHANNEL["world"]
    summed, copied = [], []
    for p in params:
        if p.grad is None:
            continue
        spec = getattr(p, "_channel_spec", None)
        if spec is None:
            copied.append(p.grad)
        else:
            summed += [p.grad.narrow(spec.dim, off, n) for off, n in spec.ranges(world, "whole")]
    _flat_collective(summed, d.all_reduce)
    _flat_collective(copied, lambda flat: d.broadcast(flat, src=0))
    return len(summed) + len(copied)


def clip_grad_norm_(parameters, max_norm):
    """torch.nn.utils.clip_grad_norm_, with the norm of the FULL gradient under fsdp-channel:
    the split parts are summed over the ranks and the whole parts counted once."""
    import torch
    ps = [p for p in parameters]
    if CHANNEL["world"] <= 1 or not dist_active():
        return torch.nn.utils.clip_grad_norm_(ps, max_norm)
    import torch.distributed as d
    world = CHANNEL["world"]
    grads = [p for p in ps if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    dev = grads[0].grad.device
    split = torch.zeros((), device=dev, dtype=torch.float32)
    whole = torch.zeros((), device=dev, dtype=torch.float32)
    for p in grads:
        g = p.grad.detach().float()
        spec = getattr(p, "_channel_spec", None)
        if spec is None:
            whole += g.square().sum()
            continue
        for off, n in spec.ranges(world, "split"):
            split += g.narrow(spec.dim, off, n).square().sum()
        for off, n in spec.ranges(world, "whole"):
            whole += g.narrow(spec.dim, off, n).square().sum()
    d.all_reduce(split)
    total = (split + whole).sqrt()
    coefficient = torch.clamp(float(max_norm) / (total + 1e-6), max=1.0)
    with torch.no_grad():
        for p in grads:
            p.grad.mul_(coefficient.to(p.grad.dtype))
    return total


def _gathered(t, spec):
    """The full tensor of `t` on the CPU, on rank 0; None on every other rank."""
    import torch
    import torch.distributed as d
    staged = t.detach().contiguous().to(_coll_device())
    parts = [torch.empty_like(staged) for _ in range(CHANNEL["world"])] if is_main() else None
    d.gather(staged, parts, dst=0)
    if not is_main():
        return None
    return spec.full([p.cpu() for p in parts])


def channel_fp32_state(model):
    """helpers.utils.fp32_state of the FULL model under fsdp-channel. Called on every rank;
    the state dict on rank 0, None on the others."""
    import torch
    owners = dict(model.named_parameters(remove_duplicate=False))
    out = {}
    for key, value in model.state_dict().items():
        spec = getattr(owners.get(key), "_channel_spec", None)
        if spec is not None:
            value = _gathered(value, spec)
            if value is None:
                continue
        elif not is_main():
            continue
        out[key] = (value.detach().float().cpu() if torch.is_tensor(value)
                    and value.is_floating_point()
                    else value.detach().cpu() if torch.is_tensor(value) else value)
    return out if is_main() else None


def channel_optimizer_state(opt):
    """The FULL MasterAdamW state dict under fsdp-channel, on the CPU. Called on every rank;
    the state dict on rank 0, None on the others."""
    import torch
    pairs = getattr(opt, "_pairs", None)
    if pairs is None:
        raise TypeError(f"fsdp-channel saves MasterAdamW state; got {type(opt).__name__}")
    sd = opt.state_dict()
    specs = [getattr(live, "_channel_spec", None) for live, _master in pairs]
    masters = []
    for spec, master in zip(specs, sd["masters"]):
        masters.append(_gathered(master, spec) if spec is not None
                       else master.detach().cpu() if is_main() else None)
    state = {}
    for key, entry in sd["inner"]["state"].items():
        spec = specs[key] if isinstance(key, int) and key < len(specs) else None
        fixed = {}
        for name, value in entry.items():
            if (spec is not None and torch.is_tensor(value) and value.dim() > 0
                    and tuple(value.shape) == tuple(pairs[key][1].shape)):
                fixed[name] = _gathered(value, spec)
            else:
                fixed[name] = value.detach().cpu() if torch.is_tensor(value) else value
        state[key] = fixed
    if not is_main():
        return None
    return {"inner": {"state": state, "param_groups": sd["inner"]["param_groups"]},
            "masters": masters}


def channel_local_optimizer_state(pairs, sd):
    """A FULL MasterAdamW state dict cut to this rank's parts, for the parameters fsdp-channel
    split; `sd` unchanged when nothing is split."""
    import torch
    specs = [getattr(live, "_channel_spec", None) for live, _master in pairs]
    if CHANNEL["world"] <= 1 or not any(specs) or len(sd["masters"]) != len(pairs):
        return sd
    rank, world = CHANNEL["rank"], CHANNEL["world"]

    def cut(value, index):
        live = pairs[index][0]
        if (specs[index] is not None and torch.is_tensor(value)
                and tuple(value.shape) == live._channel_full):
            return specs[index].local(value, rank, world)
        return value

    masters = [cut(m, i) for i, m in enumerate(sd["masters"])]
    state = {key: ({name: cut(value, key) for name, value in entry.items()}
                   if isinstance(key, int) and key < len(pairs) and isinstance(entry, dict)
                   else entry)
             for key, entry in sd["inner"]["state"].items()}
    return dict(sd, masters=masters, inner=dict(sd["inner"], state=state))


def channel_cat(t, dim):
    """`t` concatenated over the ranks along `dim` (rank order) under fsdp-channel; `t` itself
    otherwise."""
    if CHANNEL["world"] <= 1 or not dist_active():
        return t
    import torch
    import torch.distributed as d
    staged = t.detach().contiguous()
    parts = [torch.empty_like(staged) for _ in range(CHANNEL["world"])]
    d.all_gather(parts, staged)
    return torch.cat(parts, dim=dim)


def channel_mean(t):
    """The mean of `t` over the ranks under fsdp-channel; `t` itself otherwise."""
    if CHANNEL["world"] <= 1 or not dist_active():
        return t
    import torch.distributed as d
    total = t.detach().float().clone()
    d.all_reduce(total)
    return (total / CHANNEL["world"]).to(t.dtype)
