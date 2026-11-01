"""
kv_cache.py -- incremental decoding state, and the budget that bounds it.

    Cache(n_layer, budget_bytes)      an empty cache
    cache.trim(max_tokens)            drop keys/values past the window, KEEP the recurrence
    budget_bytes(gib)                the configured --kv_cache_size as bytes
    cache.fits(batch, tokens, cfg)    would decoding this batch stay inside the budget?
    plan(n_seq, tokens, cfg, budget)  how many sequences to decode at once
    bytes_per_token(cfg)              the arithmetic, exposed so a caller can size its own run
    check(model, device)              PROVE cached decoding equals uncached (see below)

WHY A CACHE AT ALL. Generation without one recomputes the whole prefix at every step: producing
n tokens costs O(n^2) forward passes' worth of work instead of O(n). At the 200-token rollouts
this pipeline used to do that was tolerable; with the response length now filling the context
window -- up to 8,000 tokens for mira-s -- it is not.

WHAT HAS TO BE CACHED DEPENDS ON THE MIXER. A conventional attention block retains K and V,
while a recurrent filter must also retain the state through which its next output depends on the
prefix. Caching keys and values while recomputing that recurrence would leave generation
quadratic anyway. A conventional recurrent-attention layer therefore keeps FOUR tensors:

    k, v     (B, heads, tokens, head_dim)   attention's keys and values, as usual
    conv     (B, d_conv - 1, C)             the depthwise causal convolution's input window
    h        (B, C)                         the recurrence state after the last token

and, under arch="mira", conv2 and h2 as well: there the queries and the keys are each produced
by their own state space system, so a layer carries two of each. MIRA-Newton packs its TWO Q/K
systems on the channel axis of conv and h -- its value is a plain per-token projection and
carries no state at all. It does not retain per-token K or V. Instead,
each head carries the mixer_k most recent positive normalizer vectors Z and matrix-valued
key--value memories N, newest first -- the mixer is ORDER mixer_k, so a step reads its own
mixer_k previous states and all of them must survive to the next step. These filter and
retrieval states are all O(1) in the sequence length and linear in mixer_k.

The recurrence states are what make MIRA steppable. The convolution keeps ``fir_m - 1`` inputs,
and each order-$N$ MIIR keeps its $N$ most recent raw states. MIRA-Newton additionally keeps the
$K$ most recent normalizer and key--value memory states. These costs grow with $N$ and $K$ but
remain constant in the generated sequence length, unlike an attention key/value cache.

THE BUDGET. For attention, the K/V part grows with every token generated, and a GRPO step decodes
batch x group_size sequences at once -- 128 of them at the default 16 x 8. Left unbounded that
is tens of gigabytes on a long window. `plan()` divides the sequences into groups that fit the
budget and the caller decodes one group at a time; the results are identical, since sequences
are independent, and only the peak changes.

    bytes = 2 (k and v) x layers x C x tokens x 4 x batch      attention, grows with tokens
          + layers x (d_conv - 1 + 1) x C x 4 x batch          recurrent filter, fixed

For MIRA-Newton the first line is zero; its two packed IIRMs and positive recurrent retrieval
contribute only fixed-size state. N, with shape (heads, mixer_k, d_h, d_v), is the largest
summary by a wide margin -- 131,072 of the 140,288 FP32 values a mira-s layer carries.

VERIFY BEFORE TRUSTING. An incremental decoder that is subtly wrong does not crash: it produces
fluent, plausible, WRONG text, and the run looks healthy while the policy learns from rollouts
the model would never have produced. `check()` runs the same prompt through both paths and
compares the logits. Run it once on the machine you train on, after any change to the model:

    python -c "import torch, helpers.kv_cache as kv, helpers.common as c; \\
               a=c.parse_args([]); ctx=c.setup(a); print(kv.check(ctx['new_model'](), ctx['device']))"
"""
import math


GIB = 1024 ** 3


def budget_bytes(size_gib):
    """A configured size in GiB as bytes. 0 or less means unbounded."""
    try:
        return max(0, int(float(size_gib) * GIB))
    except (TypeError, ValueError):
        return 0


def bytes_per_token(cfg, dtype_bytes=4):
    """Attention cache bytes for ONE token of ONE sequence: k and v, every layer."""
    return 2 * int(cfg["n_layer"]) * int(cfg["n_embd"]) * dtype_bytes


def fixed_bytes(cfg, dtype_bytes=4):
    """The SSM's part, which does not grow with the sequence: the conv window and the state."""
    d_conv, C = int(cfg.get("d_conv", 4)), int(cfg["n_embd"])
    per_layer = (max(d_conv - 1, 0) + 1) * C
    return int(cfg["n_layer"]) * per_layer * dtype_bytes


def cache_bytes(cfg, batch, tokens, dtype_bytes=4):
    """What decoding `batch` sequences to `tokens` will hold at its peak."""
    return batch * (tokens * bytes_per_token(cfg, dtype_bytes) + fixed_bytes(cfg, dtype_bytes))


def plan(n_seq, tokens, cfg, budget_bytes, dtype_bytes=4):
    """How many sequences to decode at once, and in how many groups.

    Returns (per_group, n_groups). At least one sequence always goes through: a budget too
    small for a single sequence is a budget that cannot be honoured, and refusing to decode at
    all would be worse than exceeding it -- so one sequence is decoded and the caller is told
    what it actually costs rather than silently given nothing."""
    one = cache_bytes(cfg, 1, tokens, dtype_bytes)
    per = max(1, int(budget_bytes // one)) if one else max(1, n_seq)
    per = min(per, max(1, int(n_seq)))
    return per, int(math.ceil(n_seq / per))


class Cache:
    """Per-layer decoding state. A plain container: the model reads and writes it, and nothing
    here knows what a transformer is.

    ``advance`` is called once after all layers consume a step. Architectures with per-token
    attention state also use ``append``; recurrent-only mixers instead retain constant-size
    summaries while ``tokens`` records how much input has been consumed."""

    __slots__ = ("k", "v", "conv", "h", "conv2", "h2", "mix_z", "mix_n",
                 "tokens", "budget", "key_valid")

    def __init__(self, n_layer, budget_bytes=0):
        # WHICH CACHED KEYS STAND FOR A REAL TOKEN, (B, tokens) bool, or None if no mask has
        # ever been supplied. Attention needs a mask over the WHOLE history, and a caller
        # decoding one token at a time only knows about the token it is passing -- so the
        # history is accumulated here, by MiraLLM._trunk, once per forward. Without it, a
        # prefill that carried a padding mask followed by a step that carried none would make
        # the padded prefix visible again, silently.
        self.key_valid = None
        self.k = [None] * n_layer
        self.v = [None] * n_layer
        self.conv = [None] * n_layer
        self.h = [None] * n_layer
        # A SECOND RECURRENCE PER LAYER, for arch="mira" only. There the queries and the keys
        # are each produced by their own state space system, so a layer carries two states and
        # two convolution windows instead of one. Left as None by every other architecture,
        # which never touches them and pays nothing for their existence.
        self.conv2 = [None] * n_layer
        self.h2 = [None] * n_layer
        # Positive recurrent attention carries the K MOST RECENT normalizer vectors and
        # key--value matrices per head, NEWEST FIRST: mix_z is (B, H, K, D) and mix_n is
        # (B, H, K, D, D_v). They are runtime states, not trainable parameters. The depth is K
        # because the mixer is order K -- a step reads its own K previous states -- so these
        # slots hold a stack, not a single tensor, and both grow linearly in mixer_k.
        self.mix_z = [None] * n_layer
        self.mix_n = [None] * n_layer
        self.tokens = 0
        self.budget = int(budget_bytes)

    def __len__(self):
        return self.tokens

    def append(self, layer, k, v):
        """Concatenate this step's keys and values; return the full history for attention."""
        import torch
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = torch.cat([self.k[layer], k], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v], dim=2)
        return self.k[layer], self.v[layer]

    def advance(self, n=1):
        """Count the tokens just added. Called ONCE per step by the model, after every layer."""
        self.tokens += int(n)

    def trim(self, max_tokens):
        """Drop all but the last `max_tokens` keys and values. LEAVES h AND conv ALONE.

        THE CAUSAL MASK NEEDS NO ADJUSTMENT. Attention reads the retained key length; a
        recurrent-only mixer reads ``tokens`` and carries all earlier information in its state.
        In both cases trimming the explicit mask changes no recurrent summary.

        ``tokens`` counts the explicit window after trimming; recurrent states deliberately
        continue to summarize the earlier prefix."""
        n = int(max_tokens)
        if n <= 0 or self.tokens <= n:
            return self
        if self.k[0] is not None:
            for group in (self.k, self.v):
                for i, t in enumerate(group):
                    if t is not None:
                        group[i] = t[:, :, -n:].contiguous()
        if self.key_valid is not None:
            self.key_valid = self.key_valid[:, -n:]   # it indexes the keys; it trims with them
        self.tokens = min(self.tokens, n)
        return self

    def nbytes(self):
        tot = 0
        for group in (self.k, self.v, self.conv, self.h, self.conv2, self.h2,
                      self.mix_z, self.mix_n):
            for t in group:
                if t is not None:
                    tot += t.numel() * t.element_size()
        return tot

    def over_budget(self):
        return bool(self.budget) and self.nbytes() > self.budget


def check(model, device, prompt_len=13, new_tokens=9, tol=2e-4, seed=0):
    """PROVE that decoding with the cache gives the logits decoding without it gives.

    Runs a random prompt through the model twice: once feeding the whole sequence at every
    step, once feeding one token at a time with a cache. Returns (ok, max_abs_difference).

    This is the check worth having because the failure it catches is silent. An incremental
    decoder that drops the convolution window, or forgets the recurrence state, or attends to
    one token too few, still emits fluent text -- and a GRPO run then optimises against
    rollouts the model would never have produced, with every curve looking healthy."""
    import torch
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    V = int(model.cfg["vocab_size"])
    ids = torch.randint(0, V, (2, prompt_len), generator=g).to(device)

    with torch.no_grad():
        # WITHOUT the cache: the whole sequence, every step. The reference.
        ref = []
        seq = ids
        for _ in range(new_tokens):
            logits = model(input_ids=seq).logits[:, -1]
            ref.append(logits)
            seq = torch.cat([seq, logits.argmax(-1, keepdim=True)], dim=1)

        # WITH the cache: the prompt once, then one token at a time.
        cache = Cache(len(model.blocks))
        got = []
        logits = model(input_ids=ids, cache=cache).logits[:, -1]
        got.append(logits)
        nxt = logits.argmax(-1, keepdim=True)
        for _ in range(new_tokens - 1):
            logits = model(input_ids=nxt, cache=cache).logits[:, -1]
            got.append(logits)
            nxt = logits.argmax(-1, keepdim=True)

    diff = max(float((a - b).abs().max()) for a, b in zip(ref, got))
    return diff <= tol, diff
