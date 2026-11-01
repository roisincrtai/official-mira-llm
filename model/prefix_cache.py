"""model/prefix_cache.py -- the inference-only state MIRA carries between forward passes.

WHAT MIRA HAS TO CACHE IS ONE THING PER LAYER: the persistent lattice state
V_t^(0), ..., V_t^(K-1), each of shape [d_e, d_m]. user/Notations.md gives the step

    U_t^(k-1) = U_t^(k) - diag(alpha_t^(k)) V_{t-1}^(k-1)
    V_t^(k)   = V_{t-1}^(k-1) + diag(alpha_t^(k)) U_t^(k-1),        k = K, ..., 1

and the only quantities from step t it reads are V_t^(0..K-1). Everything else is rebuilt from
the token in hand: alpha_t^(k), rho_t^(k) and gamma_t are input-conditioned, s and nu are learned
parameters rather than state, U_t^(k) is intermediate within a step, and V_t^(K) is read by the
ladder at t and never at t+1 -- which is why the persistent set is K matrices and not K+1.

There is no attention key or value to keep, no convolution window and no position table.

IT IS CONSTANT IN SEQUENCE LENGTH, which is the whole point. K * d_e * d_m floats per layer per
sequence: 98,304 at K=4, d_e=1536, d_m=16, so 384 KiB per layer and 4.5 MiB across twelve layers,
the same at 512 tokens as at 32,768. An attention cache over the same twelve layers at 512 tokens
and d_model 840 is about 41 MiB and grows linearly with the context.

TRAINING NEVER USES THIS. model.lm.train calls the model without a `cache` argument, so the
mixer receives None and this module is never constructed. That is not an accident to be tidied
into a runtime check: carrying state across training windows is truncated backpropagation through
time, it makes the loss depend on the order the batches arrive in, and it is a different objective
from the one the benchmark reports. Every training window starts from V_{-1}^(k) = 0.

THE CHUNK-BOUNDARY STATE INSIDE ONE FORWARD IS A DIFFERENT THING. The chunked scan in
model/mira_mixer.py threads a `state` tensor of exactly this shape from chunk to
chunk so the sequence is one unbroken recurrence, and recomputes each chunk in backward rather
than storing it. That state lives and dies inside a single forward pass. This one survives
between calls, is detached, and carries no gradient.
"""
from __future__ import annotations

import torch

import default_config as config

# THE DTYPE THE STATE IS CARRIED IN, AND IT IS FP32 WHATEVER THE TRUNK RUNS IN. The scan
# accumulates in float32 even under BF16 autocast -- model/mira_mixer.py casts to float before the
# recurrence -- because a state that is multiplied by a rotation at every token is where a narrow
# mantissa shows first. Storing the cache in bf16 to halve it would put the rounding back exactly
# where the model takes care to avoid it.
DTYPE = torch.float32
BYTES = 4

# GiB, SPELLED OUT ONCE. The configured budget is in GiB because that is the unit a person sizing
# a card thinks in; every comparison below is in bytes.
GIB = 1024 ** 3


def enabled():
    """Whether the prefix cache is switched on, from default_config.INFERENCE.

    READ AT CALL TIME AND NOT IMPORT TIME, because --set INFERENCE.prefix_cache=0 is applied by
    common.setup after this module may already have been imported; a value captured at import
    would ignore the flag that was passed.
    """
    return bool(int(config.INFERENCE.get("prefix_cache", 0)))


def budget_bytes():
    """The configured ceiling in bytes, or 0 when no ceiling is set."""
    return int(float(config.INFERENCE.get("prefix_cache_size", 0)) * GIB)


def state_bytes(layers, batch, order, channels, memory):
    """What one cache of this geometry occupies, in bytes.

    NOTHING HERE DEPENDS ON THE SEQUENCE LENGTH, which is the property being asserted: the same
    expression answers for a 512-token prompt and a 32,768-token one.
    """
    return int(layers) * int(batch) * int(order) * int(channels) * int(memory) * BYTES


def describe(layers, batch, order, channels, memory):
    """One line naming the geometry and what it costs, for a run to print before it decides."""
    total = state_bytes(layers, batch, order, channels, memory)
    return (f"{layers} layer(s) x batch {batch} x K={order} x d_e={channels} x d_m={memory} "
            f"= {total / GIB:.3f} GiB ({total / max(layers, 1) / 1024:.0f} KiB per layer)")


def affordable(layers, batch, order, channels, memory, budget=None):
    """(fits?, why not). A cache larger than the budget is refused rather than trimmed.

    TRIMMING IS NOT AVAILABLE HERE AND THAT IS THE POINT. An attention cache can drop its oldest
    keys and still answer, because a key belongs to one token. This state is a summary of every
    token so far folded into K matrices; there is no oldest part of it to drop, so a budget that
    does not fit is a smaller batch rather than a shorter memory.
    """
    need = state_bytes(layers, batch, order, channels, memory)
    ceiling = budget_bytes() if budget is None else int(budget)
    if ceiling and need > ceiling:
        return False, (f"the prefix cache needs {need / GIB:.3f} GiB but "
                       f"INFERENCE.prefix_cache_size allows {ceiling / GIB:.3f} GiB; "
                       f"reduce the batch -- the state does not shrink with a shorter context")
    return True, ""


class PrefixCache:
    """The persistent lattice state of every layer, carried between forward passes.

    ONE SLOT PER LAYER, HOLDING [batch, K, d_e, d_m], which is the tensor the chunked scan already
    passes from chunk to chunk. A layer that has not run yet holds None and the mixer starts it
    from zeros, which is V_{-1}^(k) = 0.

    NOTHING IS TRIMMED AND NOTHING EXPIRES. `helpers.kv_cache.Cache.trim` drops attention keys
    once they leave the window; the equivalent here would be resetting the recurrence, which
    discards the summary of everything before the window and is the one reset that must not
    happen to this architecture. mirachat.py's cached path already says so about the general case.
    """

    def __init__(self, layers, budget=None):
        self.layers = int(layers)
        self.state = [None] * self.layers
        self.budget = budget_bytes() if budget is None else int(budget)
        self.tokens = 0                     # how many tokens the state has absorbed
        self.key_valid = None

    def __len__(self):
        """HOW MANY TOKENS THE STATE HAS ABSORBED, which is what `len` means on the other cache.

        helpers.kv_cache.Cache.__len__ returns its token count, and both readers of `len(cache)`
        -- MiraLLM._trunk:656 and cot/grpo.py:192 -- use it as one. Returning the layer count
        here made those two lines mean something different depending on which cache they were
        handed, silently. The layer count is `self.layers`, which is what the constructor was
        given and what every slot list is sized by."""
        return self.tokens

    def get(self, layer):
        """The state of one layer, or None when it has not run yet."""
        return self.state[int(layer)]

    def put(self, layer, state):
        """Record one layer's state after a step. Detached: a cache carries no gradient.

        DETACHED HERE RATHER THAN AT THE CALLER, because a graph kept alive across calls is the
        failure this class must not permit: it would grow without bound over a generation and the
        second backward through it would raise on a freed graph.
        """
        self.state[int(layer)] = None if state is None else state.detach()
        return self.state[int(layer)]

    def advance(self, tokens):
        """Record that `tokens` more tokens have been folded into the state."""
        self.tokens += int(tokens)

    def clear_states(self):
        self.state = [None] * self.layers
        self.tokens = 0
        self.key_valid = None

    def reset(self, model):
        model.reset_states(self.state)
        self.tokens = 0
        self.key_valid = None

    def nbytes(self):
        """What is currently held, in bytes."""
        return sum(0 if s is None else s.numel() * s.element_size() for s in self.state)

    def within_budget(self):
        """(fits?, why not) for what is currently held."""
        held = self.nbytes()
        if self.budget and held > self.budget:
            return False, (f"the prefix cache holds {held / GIB:.3f} GiB against a "
                           f"{self.budget / GIB:.3f} GiB ceiling")
        return True, ""
