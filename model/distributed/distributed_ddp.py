"""
distributed_ddp.py -- distributed_parallelism: ddp. One process per GPU, launched by torchrun;
every rank holds the whole model and reads its own share of the batch, and the gradients are
averaged once per step. The torchrun plumbing it shares with fsdp-channel is model/parallel.py.

    distribute(model, device, log)   the model on this rank's card, rank 0's weights everywhere;
                                     nn.DataParallel over every GPU in one process
    average_grads(params)            the mean of every gradient over the ranks
"""

from model.parallel import (_REDUCE_BUCKET, broadcast_module, devices, dist_active, dist_env,
                            dist_init, report_devices)


import torch.nn as _nn


def amp_state():
    """(enabled, dtype) of the CUDA autocast on THE CALLING THREAD. Read on the main thread."""
    import torch
    try:
        on = bool(torch.is_autocast_enabled())
    except Exception:                                              # noqa: BLE001
        return (False, None)
    if not on:
        return (False, None)
    # The accessor was renamed between releases; ask for whichever this build has.
    for get, args in ((getattr(torch, "get_autocast_dtype", None), ("cuda",)),
                      (getattr(torch, "get_autocast_gpu_dtype", None), ())):
        if get is not None:
            try:
                return (True, get(*args))
            except Exception:                                      # noqa: BLE001
                pass
    return (True, None)


class _Replicated(_nn.Module):
    """One of the inner model's methods exposed as forward(), so DataParallel scatters it --
    AND RE-ENTERED INSIDE THE CALLER'S AUTOCAST, on the worker thread.

    WHY THE AUTOCAST HAS TO BE RE-ENTERED HERE.

    autocast state is THREAD-LOCAL. helpers/lm.train() opens
    `torch.autocast("cuda", dtype=torch.bfloat16)` on the MAIN thread and calls lm_loss inside
    it; nn.DataParallel then runs every replica in a FRESH threading.Thread, and
    torch.nn.parallel.parallel_apply re-enters autocast in that thread carrying `enabled` but
    NOT `dtype`. The worker therefore falls back to the CUDA autocast default, which is fp16.

    SO THE TRUNK RAN fp16 AND THE HEAD RAN bf16, on multi-GPU only. helpers/lm.lm_loss calls
    model.head on the MAIN thread, outside DataParallel, so the vocabulary projection and the
    chunked log-softmax kept the intended bf16 while all twenty blocks inside the wrapper did
    not -- and this pipeline has NO GradScaler, deliberately (see helpers/utils.amp: "bf16 keeps
    fp32's eight exponent bits ... NO LOSS SCALING IS NEEDED. fp16 needs a GradScaler, which is
    a second mechanism to misconfigure; this pipeline has none and needs none").

    WHAT THAT COST, measured: 4xA100 held next-token accuracy at 0.044 for 800 steps while loss
    fell to 8.2; one A100 on the same commit tracked one L40S step for step (0.056 vs 0.057 at
    step 8) and kept climbing. The head alone can fit the unigram marginal -- which is what a
    falling loss with a frozen argmax IS -- and the tied embedding/head is 24% of the
    parameters, which is why the gradient norm stayed near 1 instead of collapsing.

    THE CAPTURE IS MASTER -> REPLICA, WHICH IS THE DIRECTION THAT WORKS. `_amp` is set on this
    module on the main thread before dispatch; replicate() copies __dict__ into each replica, so
    every worker reads it. (The REVERSE direction is the one that silently discards writes --
    see model/mira._Probed, which exists for exactly that.)"""

    def __init__(self, inner, method=None):
        super().__init__()
        self.inner = inner
        object.__setattr__(self, "_method", method)
        object.__setattr__(self, "_amp", (False, None))

    def forward(self, *a, **kw):
        import torch
        fn = self.inner if self._method is None else getattr(self.inner, self._method)
        on, dt = self._amp
        if not on or dt is None:
            return fn(*a, **kw)
        with torch.autocast("cuda", dtype=dt):
            return fn(*a, **kw)


class DataParallelModel(_nn.Module):
    """A MiraLLM replicated across `device_ids`, that still behaves like a MiraLLM.

    Attribute reads and writes pass through to the inner model, so `model.cfg`, `model.head`,
    `model.grad_checkpoint`, `model.ssm`, `model.mira` and the diagnostics all work unchanged --
    and so does assigning to them. `.module` is the real model when it is needed explicitly."""

    _OWN = ("module",)

    def __init__(self, model, device_ids):
        super().__init__()
        self.module = model
        ids = list(device_ids)
        # NOT registered as submodules: they hold the SAME parameters as `module`, and
        # registering them would have the optimiser and every parameter count see them twice.
        object.__setattr__(self, "_ids", ids)
        # BOTH PATHS GO THROUGH _Replicated, so both re-enter the caller's autocast dtype on the
        # worker thread. _fwd wraps forward(), _hid wraps hidden_states() -- the one the chunked
        # loss actually uses, and therefore the one the training step runs.
        object.__setattr__(self, "_fwd_m", _Replicated(model))
        object.__setattr__(self, "_hid_m", _Replicated(model, "hidden_states"))
        object.__setattr__(self, "_fwd", _nn.DataParallel(self._fwd_m, device_ids=ids))
        object.__setattr__(self, "_hid", _nn.DataParallel(self._hid_m, device_ids=ids))

    @property
    def device_ids(self):
        return list(self._ids)

    # THE AUTOCAST IS READ HERE, ON THE MAIN THREAD, WHERE IT IS CORRECT, and handed to the
    # replicated module before dispatch. Reading it inside the worker is what was broken.
    def forward(self, *a, **kw):
        self._fwd_m._amp = amp_state()
        return self._fwd(*a, **kw)

    def hidden_states(self, *a, **kw):
        self._hid_m._amp = amp_state()
        return self._hid(*a, **kw)

    # THE CHECKPOINT IS THE SAME FILE WITH OR WITHOUT THIS WRAPPER. See point 2 above.
    def state_dict(self, *a, **kw):
        return self.module.state_dict(*a, **kw)

    def load_state_dict(self, *a, **kw):
        return self.module.load_state_dict(*a, **kw)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["module"], name)

    def __setattr__(self, name, value):
        inner = self.__dict__.get("_modules", {}).get("module")
        if inner is not None and name not in self._OWN and not name.startswith("_") \
                and hasattr(inner, name):
            return setattr(inner, name, value)          # see point 3 above
        return super().__setattr__(name, value)


def data_parallel(model, device_list=None, log=print):
    # DDP FIRST WHEN THE LAUNCHER PROVIDED ONE. Under torchrun there is one process per GPU
    # and nothing to scatter: each rank already holds a replica and draws its own shard, so
    # only the gradients have to be all-reduced. Falling through to nn.DataParallel here would
    # put N replicas on one device and gather on it.
    _rank, _local, _world = dist_env()
    if _world > 1:
        dist_init(log)
        # THE MODEL ITSELF, ON THIS RANK'S CARD, WITH NOTHING WRAPPED AROUND IT. `model.cfg`,
        # `model.head`, `model.hidden_states` and every state_dict key are then exactly what
        # they are on one GPU -- no "module." prefix -- so a checkpoint written under torchrun
        # loads anywhere and one written on a laptop resumes under four ranks. The gradients
        # are averaged explicitly after the backward instead; see all_reduce_grads.
        import torch
        from kernel.hardware import resolve
        import os
        selected = resolve(os.environ.get("GPU", "auto"))
        model = model.to(torch.device(selected.type, _local) if selected.type in ("cuda", "npu") else selected)
        # AND THE ONE THING DistributedDataParallel'S CONSTRUCTOR DID THAT STILL HAS TO HAPPEN.
        # See broadcast_module: nothing here seeds torch's global generator, so without this
        # the four ranks hold four different random initialisations.
        broadcast_module(model)
        if _rank == 0:
            # SAID UNDER DDP TOO. report_devices is the answer to "how many cards did this
            # process actually get", and returning before it left every torchrun run with no
            # [gpu] line at all -- on the launcher that makes the count least obvious.
            report_devices(log)
            log(f"[parallel] DDP over {_world} rank(s)")
        return model

    """Replicate `model` across every visible CUDA device. Returns it unchanged when there is
    one device or none -- a single-GPU machine takes the same path and nothing is wrapped.

    THE PARAMETER IS NOT CALLED `devices`, AND THAT IS THE POINT. It was, and a parameter of
    that name SHADOWS the module-level devices() function inside this body, so the default path
    -- the only one the trainer uses -- evaluated `devices()` with `devices` bound to None and
    died with "'NoneType' object is not callable" on the first build_model of every run. The
    name is different now so the two cannot collide again."""
    devs = list(device_list) if device_list is not None else devices()
    report_devices(log)
    if len(devs) < 2:
        return model
    if devs[0].type == "npu":
        raise RuntimeError("Ascend DDP requires one torchrun process per NPU")
    ids = [int(str(d).split(":")[1]) if ":" in str(d) else int(d) for d in devs]
    log(f"[parallel] data parallel across {len(ids)} GPUs {ids}")
    return DataParallelModel(model, ids)


def distribute(model, device=None, log=print):
    """distributed_parallelism: ddp."""
    if device is not None:
        model = model.to(device)
    return data_parallel(model, log=log)


def average_grads(params):
    """Average every parameter's `.grad` across the ranks, in place. Returns how many tensors
    were reduced; a no-op returning 0 off DDP, where there is nothing to average.

    CALLED ONCE PER OPTIMISER STEP, after the last micro-batch's backward and BEFORE the
    gradient clipping and the step -- so the norm that is clipped, the update that is applied
    and the number the log reports are all the whole batch's, identically on every rank.

    THE MEAN IS THE RIGHT REDUCTION because each rank has already weighted its micro-batches by
    their share of ITS OWN batch (helpers/lm.train), so each rank's gradient is the mean over
    the per-GPU batch it saw. Every rank draws the same configured batch, so the mean of the
    ranks' means is the mean over the whole global batch,
    which is the gradient a single GPU would have computed for that batch.

    A PARAMETER WITH NO GRADIENT KEEPS NONE, AND THE DECISION IS MADE COLLECTIVELY. `.grad is
    None` is a local fact; acting on it locally is how a hang is written. So the presence flags
    are all-reduced FIRST, in one fixed-size message, and every rank then reduces exactly the
    same set of tensors in the same order: a parameter that no rank has a gradient for is
    skipped everywhere and stays None, and one that SOME rank has is given a zero gradient on
    the ranks that lack it so that the sum is over the same tensors on all of them. Off the
    equal-work path this pipeline actually runs, that is what keeps the collective count equal
    rather than merely equal in practice.

    BUCKETED, NOT ONE MESSAGE PER TENSOR AND NOT ONE MESSAGE FOR THE WHOLE MODEL. Per tensor is
    hundreds of small NCCL calls a step; one flat buffer over every gradient is a second fp32
    copy of the model, which at mira-l is 15 GiB the card does not have. The buckets are cut
    by element count from a list that is identical on every rank, so the number of messages and
    their order are identical too."""
    if not dist_active():
        return 0
    import torch
    import torch.distributed as d
    ps = list(params)
    if not ps:
        return 0
    world = d.get_world_size()
    # THE FLAGS GO ON A PARAMETER'S OWN DEVICE, which under torchrun is this rank's card -- NCCL
    # cannot reduce a CPU tensor, and "cuda" here would be right only because dist_init already
    # pinned it. Taking the device from the model asks nothing of the environment.
    dev = ps[0].device
    have = torch.tensor([0.0 if p.grad is None else 1.0 for p in ps],
                        device=dev, dtype=torch.float32)
    d.all_reduce(have, op=d.ReduceOp.SUM)
    flags = have.tolist()
    grads = []
    with torch.no_grad():
        for p, f in zip(ps, flags):
            if f <= 0.0:                      # no rank produced one: it stays None everywhere
                continue
            if p.grad is None:
                p.grad = torch.zeros_like(p)  # some rank did: contribute zero to the sum
            grads.append(p.grad)
    if not grads:
        return 0
    buckets, cur, cur_n = [], [], 0
    for g in grads:
        cur.append(g)
        cur_n += g.numel()
        if cur_n >= _REDUCE_BUCKET:
            buckets.append(cur)
            cur, cur_n = [], 0
    if cur:
        buckets.append(cur)
    with torch.no_grad():
        for b in buckets:
            # fp32 FOR THE REDUCTION, whatever the gradients are stored in. bf16 has three
            # significant digits, and summing four of them before dividing is exactly where
            # that is not enough; the copy back returns each gradient to its own dtype.
            flat = torch.cat([g.detach().reshape(-1).float() for g in b])
            d.all_reduce(flat, op=d.ReduceOp.SUM)
            flat /= world
            off = 0
            for g in b:
                n = g.numel()
                g.copy_(flat[off:off + n].view_as(g))
                off += n
    return len(grads)
