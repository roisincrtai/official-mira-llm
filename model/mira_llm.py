import importlib
import math

import torch
import torch.nn as nn

_matrix_arch = importlib.import_module("model.mira_mixer")
_mira_block = importlib.import_module("model.mira_block")
MiraBlock = _mira_block.MiraBlock
RMSNorm = _mira_block.RMSNorm


class Output:
    """Minimal stand-in for a HF model output (.logits, plus .mask_logit when requested)."""
    def __init__(self, logits, mask_logit=None):
        self.logits = logits
        self.mask_logit = mask_logit


class MiraLLM(nn.Module):
    def __init__(self, vocab_size, n_layer=12, n_head=None, n_embd=512, block_size=512,
                 dropout=0.0, ffn_factor=None, gated_attn=True, d_conv=4, ssm_chunk="optimal",
                 lattice_modulus_map=None, lattice_argument_map=None, fir_m=4, iir_n=2, ffn_hidden=None, ffn_input_bias=True,
                 miir_silu=False, qk_rmsnorm=False, qk_coupling=True, mixer_k=2,
                 lattice_order=None, lattice_iir_order=None,
                 lattice_d_e=None, coefficient_bottleneck=None,
                 lattice_expansion_factor=None,
                 coefficient_bottleneck_factor=None,
                 memory_dimension=None,
                 lattice_output_gate=True,
                 lattice_positive_poles=False,
                 lattice_initializer="W_uniform_b_jeffreys",
                 lattice_init_horizon=None,
                 lattice_init_method=None,
                 moe_total_experts=1, moe_layers=None, moe_active_experts=None,
                 moe_expert_width=None, moe_shared_experts=0):
        super().__init__()
        import default_config as _config
        # THE FFN WIDTH IS ffn_hidden, the scheme's ffn_width, and 4 x n_embd when there is none
        # (a checkpoint or caller that does not record it). ffn_factor is accepted because older
        # checkpoints record it, and it sizes nothing.
        del ffn_factor
        ffn_hidden = int(ffn_hidden) if ffn_hidden else 4 * int(n_embd)
        lattice_modulus_map = (lattice_modulus_map if lattice_modulus_map is not None
                               else _config.MODEL["lattice_modulus_map"])
        lattice_argument_map = (lattice_argument_map if lattice_argument_map is not None
                                else _config.MODEL["lattice_argument_map"])
        lattice_arch = _config.mira_map_arch({
            "lattice_modulus_map": lattice_modulus_map,
            "lattice_argument_map": lattice_argument_map})
        if lattice_positive_poles and lattice_arch != "cauchy_x_mobius":
            raise ValueError("lattice_positive_poles requires lattice_modulus_map=cauchy and lattice_argument_map=mobius")
        if not lattice_output_gate:
            raise ValueError("lattice_output_gate must be true")
        # THE GEOMETRY'S NUMBERS LIVE IN THE CONFIG AND NOWHERE ELSE. A caller that does not
        # name one takes the configured value; the architecture module holds no copy of it.
        if lattice_order is None:
            lattice_order = _config.MODEL["lattice_order"]
        if lattice_iir_order is None:
            lattice_iir_order = lattice_order
        if lattice_expansion_factor is None:
            lattice_expansion_factor = _config.MODEL["lattice_expansion_factor"]
        if coefficient_bottleneck_factor is None:
            coefficient_bottleneck_factor = _config.MODEL["coefficient_bottleneck_factor"]
        if memory_dimension is None:
            memory_dimension = _config.MODEL["memory_dimension"]
        if int(lattice_expansion_factor) < 1:
            raise ValueError("lattice_expansion_factor must be positive")
        if int(coefficient_bottleneck_factor) < 1:
            raise ValueError("coefficient_bottleneck_factor must be positive")
        if lattice_d_e is None:
            lattice_d_e = int(n_embd) * int(lattice_expansion_factor)
        if coefficient_bottleneck is None:
            if int(n_embd) % int(coefficient_bottleneck_factor):
                raise ValueError("n_embd must be divisible by coefficient_bottleneck_factor")
            coefficient_bottleneck = int(n_embd) // int(coefficient_bottleneck_factor)
        if int(lattice_d_e) < 1:
            raise ValueError("lattice_d_e must be positive")
        if int(coefficient_bottleneck) < 1:
            raise ValueError("coefficient_bottleneck must be positive")
        if int(memory_dimension) < 1:
            raise ValueError("memory_dimension must be positive")
        if lattice_positive_poles and int(lattice_order) != 1:
            raise ValueError("lattice_positive_poles requires lattice_order=1")
        # lattice_init_method IS THE NAME BEFORE 2026-10-05; a caller that still passes it is honoured.
        if lattice_init_method is not None:
            lattice_initializer = lattice_init_method
        lattice_initializer = _matrix_arch.parse_lattice_initializer(
            lattice_initializer)
        lattice_init_horizon = _matrix_arch.parse_lattice_init_horizon(
            lattice_init_horizon)
        self.cfg = dict(
            lattice_modulus_map=lattice_modulus_map,
            lattice_argument_map=lattice_argument_map,
            vocab_size=vocab_size,
            n_layer=int(n_layer),
            n_embd=int(n_embd),
            lattice_order=int(lattice_order),
            lattice_d_e=int(lattice_d_e),
            coefficient_bottleneck=int(coefficient_bottleneck),
            memory_dimension=int(memory_dimension),
            ffn_hidden=None if ffn_hidden is None else int(ffn_hidden),
            ffn_input_bias=bool(ffn_input_bias),
            lattice_initializer=str(lattice_initializer),
            lattice_init_horizon=(dict(lattice_init_horizon) if isinstance(lattice_init_horizon, dict)
                                  else tuple(lattice_init_horizon)))
        experts = int(moe_total_experts or 1)
        moe, moe_span = None, (1, 0)
        if experts > 1:
            if moe_active_experts is None or moe_expert_width is None:
                raise ValueError("moe_total_experts > 1 needs moe_active_experts and "
                                 "moe_expert_width")
            moe_span = (tuple(int(n) for n in moe_layers) if moe_layers is not None
                        else (0, int(n_layer) - 1))
            if len(moe_span) != 2 or not 0 <= moe_span[0] <= moe_span[1] < int(n_layer):
                raise ValueError(f"moe_layers must be [first, last] within {int(n_layer)} "
                                 f"layers; got {moe_layers}")
            moe = dict(experts=experts, active=int(moe_active_experts),
                       expert_width=int(moe_expert_width),
                       shared=int(moe_shared_experts or 0))
            self.cfg.update(moe_total_experts=experts, moe_layers=list(moe_span),
                            moe_active_experts=moe["active"],
                            moe_expert_width=moe["expert_width"],
                            moe_shared_experts=moe["shared"])
        self.block_size = block_size
        self.grad_checkpoint = False
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.mask_embed = nn.Parameter(torch.zeros(n_embd))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            MiraBlock(n_embd, dropout=dropout, lattice_order=int(lattice_order),
                               d_e=int(lattice_d_e),
                               coefficient_bottleneck=int(coefficient_bottleneck),
                               memory_dimension=int(memory_dimension),
                               positive_poles=bool(lattice_positive_poles),
                               lattice_initializer=str(lattice_initializer),
                               lattice_init_horizon=_matrix_arch.mira_initializer.layer_lattice_init_horizon(
                                   lattice_init_horizon, layer),
                               ffn_hidden=ffn_hidden, arch=lattice_arch,
                               moe=moe if moe_span[0] <= layer <= moe_span[1] else None)
            for layer in range(n_layer)
        ])
        dense = next((block.ffn for block in self.blocks
                      if isinstance(block.ffn, _mira_block.FeedForward)), None)
        if dense is not None:
            self.cfg["ffn_hidden"] = int(dense.up.out_features)
            self.cfg["ffn_input_bias"] = dense.up.bias is not None
        else:
            self.cfg["ffn_hidden"] = int(ffn_hidden)
            self.cfg["ffn_input_bias"] = True
        self.lnf = RMSNorm(n_embd, eps=1e-8, bias=True)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.tok.weight = self.head.weight
        self.apply(self._init)
        self._resize_note = ""
        for blk in self.blocks:
            blk.post_init(n_layer)

    def resize_token_embeddings(self, new_vocab_size, seed=0, allow_shrink=False):
        """Grow the vocabulary to `new_vocab_size`, KEEPING every existing row.

        This is the model half of registering a special token. The tokenizer appends the new
        id after everything already learned, so row i of the embedding still means token i for
        every i that existed before; the matrix simply needs more rows. Fine-tuning a
        pretrained checkpoint on a corpus that uses <xyz> therefore costs one new embedding
        row, not a retrained model.

        NEW ROWS ARE INITIALISED AT THE MEAN of the existing embeddings plus a little noise,
        rather than at zero or at N(0, 0.02). The mean is where a token of no particular
        meaning belongs: it starts out maximally unopinionated instead of being a large
        outlier the first gradients have to drag back into the cloud. The noise breaks the
        symmetry between several tokens registered at once, and is scaled to the spread of the
        rows already there, so it does not depend on the width of the model. HuggingFace's
        mean_resizing does the same thing for the same reason.

        Weight tying is re-established explicitly: the embedding and the output head are ONE
        parameter here, and a resize that rebuilt them separately would silently untie them --
        the model would still run, and would train two copies of a matrix the architecture
        says is one.
        """
        old = self.tok.weight.shape[0]
        if new_vocab_size == old:
            return self
        if new_vocab_size < old and not allow_shrink:
            # allow_shrink is for ONE caller: sizing a freshly built model down to match a
            # checkpoint before loading it. Nothing is lost there because nothing is trained
            # yet. Shrinking a trained model is refused, because it can only mean the
            # vocabulary was rebuilt.
            raise ValueError(
                f"refusing to shrink the vocabulary {old} -> {new_vocab_size}: dropping rows "
                f"would change what the surviving ids mean only if the tokenizer also "
                f"changed, in which case this checkpoint belongs to a different vocabulary.")
        w = self.tok.weight.data
        keep = min(old, new_vocab_size)
        grown = w.new_empty((new_vocab_size, w.shape[1]))
        grown[:keep] = w[:keep]
        if new_vocab_size > old:
            mu, sd = w.mean(0, keepdim=True), w.std().item()
            g = torch.Generator(device="cpu").manual_seed(seed)
            fresh = (mu.cpu().repeat(new_vocab_size - old, 1)
                     + 0.02 * sd * torch.randn(new_vocab_size - old, w.shape[1], generator=g))
            grown[old:] = fresh.to(w.device, w.dtype)
        self.tok.weight.data = grown
        self.tok.num_embeddings = new_vocab_size
        self.head.out_features = new_vocab_size
        # ALREADY TIED, AND STILL TIED. Stated rather than assumed: if a caller ever unties them
        # this is what puts the head back on the embedding's rows, and it is a no-op otherwise.
        self.head.weight = self.tok.weight
        self.cfg["vocab_size"] = new_vocab_size
        self._resize_note = (
            f"vocabulary {old} -> {new_vocab_size}: {new_vocab_size - old} row(s) added at "
            f"the embedding mean" if new_vocab_size > old else
            f"vocabulary {old} -> {new_vocab_size}")
        return self

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _trunk(self, input_ids, attention_mask, mask_positions, mask_gate, mask_vec,
               sub_ids=None, cache=None):
        # THE INPUT FOLLOWS THE EMBEDDING, not the other way round. A caller hands ids on
        # whatever device it built them on; the lookup has to happen where the (tied) embedding
        # matrix is. Unsharded that is a no-op, since the two already agree.
        home = self.tok.weight.device
        if input_ids is not None and input_ids.device != home:
            input_ids = input_ids.to(home)
            if attention_mask is not None:
                attention_mask = attention_mask.to(home)
            if mask_positions is not None:
                mask_positions = mask_positions.to(home)
            if mask_gate is not None:
                mask_gate = mask_gate.to(home)
            if sub_ids is not None:
                sub_ids = sub_ids.to(home)
        emb = self.tok(input_ids)
        tgt = self.tok(sub_ids) if sub_ids is not None else (
            self.mask_embed if mask_vec is None else mask_vec)
        if mask_positions is not None:                       # hard boolean replace
            emb = torch.where(mask_positions.unsqueeze(-1), tgt.to(emb.dtype), emb)
        if mask_gate is not None:                            # differentiable gate
            g = mask_gate.unsqueeze(-1).to(emb.dtype)        # (B,T,1) in [0,1]
            emb = emb + g * (tgt.to(emb.dtype) - emb)        # g=1 -> substitute/MASK
        x = self.drop(emb)                                   # no position added here: see Block
        # WHEN THE BLOCKS ARE SPREAD ACROSS DEVICES, the hidden states follow them. One
        # (B, T, n_embd) copy per boundary and nothing else changes: the blocks do not know
        # where they are, the arithmetic is identical, and an unsharded model takes the same
        # path with `devs` empty and every comparison false. See model/distributed/distributed_fsdp_layer.py.
        # THE HISTORY OF WHICH KEYS ARE REAL LIVES IN THE CACHE, NOT IN THE ARGUMENT. Attention
        # masks the whole key history; a caller decoding one token at a time knows only about
        # the token it is passing. Resolving that here, once per forward, does three things a
        # block cannot: the mask a block sees is always the full history, so its length contract
        # is unambiguous; a step that passes no mask after a padded prefill does not make the
        # padding visible again; and the accumulation happens once rather than per layer.
        if cache is not None:
            cur = attention_mask[:, -x.shape[1]:] if attention_mask is not None else None
            if cur is not None or getattr(cache, "key_valid", None) is not None:
                if cur is None:                       # no mask this step: the new tokens are real
                    cur = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
                cur = cur.bool()
                prev = cache.key_valid
                if prev is None:
                    # A MASK ARRIVING LATE STILL HAS TO ACCOUNT FOR WHAT IS ALREADY CACHED.
                    # Decoding can begin unmasked -- key_valid stays None -- and a caller may
                    # then supply one. Storing just this step's mask would give a history of
                    # length T against keys of length past+T: the block raises, correctly, but
                    # only after key_valid has been written, leaving the cache describing
                    # itself wrongly and the next call raising for a different reason. The
                    # already-cached prefix was decoded unmasked, so by construction it is
                    # real, and is recorded as such before this step's mask is appended.
                    past = len(cache)
                    if past:
                        prev = torch.ones(x.shape[0], past, dtype=torch.bool, device=x.device)
                cache.key_valid = cur if prev is None else torch.cat([prev, cur], dim=1)
                attention_mask = cache.key_valid
        # WHETHER THE BLOCKS ARE CHECKPOINTED, decided once for the whole trunk.
        #
        # WHY ALL THREE CONDITIONS. Checkpointing stores a block's INPUT and recomputes its
        # interior during the backward pass, so it is only a saving when there IS a backward
        # pass: under no_grad the recomputation would never happen and the wrapper is pure
        # overhead. And it must be OFF whenever a cache is live, because the recomputation
        # re-runs the block -- which would append to the KV cache a second time and advance
        # both recurrences twice for one token. Generation and evaluation therefore never
        # checkpoint, which is also when memory is not the binding constraint.
        ckpt = bool(getattr(self, "grad_checkpoint", False) and self.training
                    and torch.is_grad_enabled() and cache is None)
        run = self._ckpt_block if ckpt else (lambda blk, *a: blk(*a))
        devs = getattr(self, "_shard_devices", None)
        if not devs:
            for i, blk in enumerate(self.blocks):
                x = run(blk, x, attention_mask, cache, i)
            if cache is not None:
                cache.advance(x.shape[1])
            return self.lnf(x)
        at, mask = None, attention_mask
        for j, (blk, i) in enumerate(zip(self.blocks, self._shard_where)):
            if devs[i] != at:
                at = devs[i]
                x = x.to(at)
                mask = attention_mask.to(at) if attention_mask is not None else None
            x = run(blk, x, mask, cache, j)
        if cache is not None:
            cache.advance(x.shape[1])
        return self.lnf(x.to(devs[-1]))

    def reset_states(self, lattice_states=None):
        if lattice_states is not None:
            for layer in range(len(lattice_states)):
                lattice_states[layer] = None
        return None

    @staticmethod
    def _ckpt_block(blk, x, mask, cache, layer):
        """One block, with its interior recomputed in the backward pass instead of stored.

        use_reentrant=False, and not as a default-following gesture. The reentrant
        implementation runs the block under its own autograd graph, which breaks when the
        block's output does not require grad, ignores keyword arguments, and does not compose
        with the autocast region this runs inside. The non-reentrant one hooks the saved
        tensors instead, so it is transparent to all three.

        RNG STATE IS PRESERVED, which is the difference between checkpointing and silently
        training a different model: dropout draws a mask in the forward, and if the
        recomputation drew a DIFFERENT one, the gradient would be the gradient of a network
        that never produced the loss. preserve_rng_state defaults True and is passed
        explicitly, because it is the correctness argument and not an incidental default."""
        own = getattr(blk, "channel_checkpointed", None)
        if own is not None:
            return own(x, mask, cache, layer)
        from torch.utils.checkpoint import checkpoint
        return checkpoint(blk, x, mask, cache, layer,
                          use_reentrant=False, preserve_rng_state=True)

    def hidden(self, input_ids=None, attention_mask=None, mask_positions=None, mask_gate=None,
               mask_vec=None, sub_ids=None, cache=None, **kw):
        """Final-layer hidden states (B,T,n_embd) -- what a linear mask head reads."""
        return self._trunk(input_ids, attention_mask, mask_positions, mask_gate, mask_vec,
                           sub_ids, cache)

    def forward(self, input_ids=None, attention_mask=None, mask_positions=None, mask_gate=None,
                mask_vec=None, sub_ids=None, mask_logits=False, cache=None, **kw):
        """`sub_ids` makes the gate interpolate toward substitute-token embeddings (resampling
        corruption); otherwise it interpolates toward the MASK embedding."""
        x = self._trunk(input_ids, attention_mask, mask_positions, mask_gate, mask_vec,
                        sub_ids, cache)
        ml = (x @ self.mask_embed.to(x.dtype)) if mask_logits else None
        return Output(self.head(x), ml)

    def hidden_states(self, input_ids=None, attention_mask=None, mask_positions=None,
                      mask_gate=None, mask_vec=None, sub_ids=None, cache=None, **kw):
        """The trunk's output, (B, T, n_embd), BEFORE the projection to the vocabulary.

        forward() applies `head` to all T positions at once, which produces a (B, T, vocab)
        tensor -- 6.1 GB at T = 32,768 for a 50,259-token vocabulary, and the loss needs three
        of them. A caller that intends to project a slice at a time asks for the hidden states
        instead and applies `head` itself; see model/lm.py. Nothing else differs, so the two
        routes give identical logits."""
        return self._trunk(input_ids, attention_mask, mask_positions, mask_gate, mask_vec,
                           sub_ids, cache)
