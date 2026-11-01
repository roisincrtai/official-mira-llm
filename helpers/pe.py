"""
pe.py -- positional encodings.

    RotaryPositionalEmbedding     -- RoPE (Su et al. 2021), used by mira_llm; rotates the per-head
                                     query/key vectors inside attention rather than adding to the
                                     token embeddings, so there is no position table and the model
                                     extrapolates to any length. Carries YaRN (Peng et al. 2023)
                                     as an evaluation-time reconfiguration -- see configure_yarn.
    SinusoidalPositionalEncoding  -- fixed sin/cos table (Vaswani et al.)
    LearnedPositionalEncoding     -- trainable per-position embedding (GPT-2 style)

The additive encodings are cache-aware: forward(x, start=k) adds the encodings for absolute
positions [k, k+T). RoPE takes (q, k) and returns them rotated by the same absolute positions.
"""
import math

import torch
import torch.nn as nn


# YaRN's two boundaries, in ROTATIONS OVER THE TRAINING CONTEXT, and they are the paper's own.
#
# A RoPE pair of wavelength lambda completes L/lambda rotations inside a training context of L
# tokens. YaRN's observation is that the two ends of the spectrum must be treated oppositely:
#
#   more than BETA_FAST rotations   the pair has been seen at every phase during training, so its
#                                   frequency is left ALONE and simply extrapolated
#   fewer than BETA_SLOW rotations  the pair never completed a turn, so extending the context
#                                   would take it into phases it has never seen; its frequency is
#                                   INTERPOLATED by the full scale
#   between the two                 a linear ramp between those, which is what "by parts" means
#
# 32 and 1 are the values Peng et al. report for Llama and are what every implementation uses.
YARN_BETA_FAST, YARN_BETA_SLOW = 32.0, 1.0


def _yarn_correction_dim(rotations, head_dim, base, original_context):
    """The PAIR INDEX whose wavelength completes `rotations` turns in `original_context` tokens.

    inv_freq_i = base^(-2i/head_dim), so that pair's wavelength is 2*pi*base^(2i/head_dim) and
    L / lambda_i = r solves to i = head_dim * log(L / (2*pi*r)) / (2*log(base)). The index is
    real-valued on purpose: the ramp between the two boundaries is continuous."""
    return (head_dim * math.log(original_context / (rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_correction_range(beta_fast, beta_slow, head_dim, base, original_context):
    """(low, high) pair indices bounding the ramp. low comes from the FAST boundary: the
    correction dimension falls as the rotation count rises, so 32 rotations names a smaller
    index than 1 does."""
    low = math.floor(_yarn_correction_dim(beta_fast, head_dim, base, original_context))
    high = math.ceil(_yarn_correction_dim(beta_slow, head_dim, base, original_context))
    return max(low, 0), min(high, head_dim - 1)


def _yarn_ramp(low, high, pairs):
    """0 below `low`, 1 above `high`, linear between. Guarded where the two coincide, which
    happens at a short training context where floor and ceil land on the same index."""
    if low == high:
        high = high + 0.001
    ramp = (torch.arange(pairs, dtype=torch.float32) - low) / (high - low)
    return ramp.clamp(0.0, 1.0)


class RotaryPositionalEmbedding(nn.Module):
    """RoPE applied to per-head query/key tensors of shape (B, n_head, T, head_dim). `head_dim`
    must be even. No learned parameters; positions are encoded by rotation.

    YaRN IS A RECONFIGURATION OF THIS MODULE, NOT A SECOND ONE. `configure_yarn` replaces the
    frequency table and sets an attention scale; `clear_yarn` puts both back. Nothing is learned
    either way and no parameter of the model moves, which is the point: the same trained weights
    are asked the same question through a different frequency table, so a difference in the
    result belongs to the extension method and to nothing else.
    """
    def __init__(self, head_dim, base=10000):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE needs an even head dimension"
        self.head_dim = int(head_dim)
        self.base = float(base)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # TWO BUFFERS, AND THE TRAINED ONE IS NEVER WRITTEN. `inv_freq` is what forward reads and
        # is what YaRN replaces; `base_inv_freq` is the table the model was trained under, kept
        # so clear_yarn restores it exactly rather than recomputing and hoping for the same
        # floating-point result. Neither is persistent: both are derived from head_dim and base,
        # so a checkpoint carries no rotary state and this change adds no key to any state dict.
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("base_inv_freq", inv_freq.clone(), persistent=False)
        # THE ATTENTION SCALE, 1.0 UNTIL YaRN SETS IT. Folded into cos and sin rather than
        # applied to the logits: forward returns m*R(q) and m*R(k), so the dot product carries
        # m^2, which is YaRN's 1/t. Doing it here keeps every attention implementation -- eager,
        # SDPA, a fused kernel -- scaled identically without any of them knowing.
        self.attention_scale = 1.0
        self.yarn = None                      # the settings in force, or None; read by reports

    def configure_yarn(self, scale, original_context,
                       beta_fast=YARN_BETA_FAST, beta_slow=YARN_BETA_SLOW,
                       attention_scale=None):
        """Switch this module to YaRN at `scale` = target context / `original_context`.

        YaRN (Peng et al. 2023, arXiv:2309.00071) is NTK-BY-PARTS INTERPOLATION plus an
        ATTENTION TEMPERATURE, and both halves are here.

        BY PARTS. Position interpolation divides every frequency by the scale, which crushes the
        high-frequency pairs that carry local order; pure NTK scaling of the base leaves the
        low-frequency pairs extrapolating into phases never trained. YaRN interpolates only the
        pairs that need it, chosen by how many rotations each completes in the training context,
        with a linear ramp between the two boundaries.

        THE TEMPERATURE. Lengthening the context flattens the attention distribution -- more
        keys, same logit scale -- so YaRN divides the logits by t with sqrt(1/t) = 0.1*ln(s) + 1,
        an empirical fit the paper gives and the one every implementation uses. It is folded into
        cos and sin here; see the note on attention_scale.

        A scale of 1 or less is not an extension and clears instead, so a sweep may call this at
        every length without deciding which lengths are extrapolation.
        """
        scale = float(scale)
        if scale <= 1.0:
            return self.clear_yarn()
        original_context = int(original_context)
        extrapolation = self.base_inv_freq
        interpolation = self.base_inv_freq / scale
        low, high = _yarn_correction_range(beta_fast, beta_slow, self.head_dim, self.base,
                                           original_context)
        keep = (1.0 - _yarn_ramp(low, high, self.head_dim // 2)).to(extrapolation)
        self.inv_freq = interpolation * (1.0 - keep) + extrapolation * keep
        self.attention_scale = (float(attention_scale) if attention_scale is not None
                                else 0.1 * math.log(scale) + 1.0)
        self.yarn = {"scale": scale, "original_context": original_context,
                     "beta_fast": float(beta_fast), "beta_slow": float(beta_slow),
                     "low": int(low), "high": int(high),
                     "attention_scale": float(self.attention_scale)}
        return self.yarn

    def clear_yarn(self):
        """Back to the frequencies and the logit scale the model was trained under."""
        self.inv_freq = self.base_inv_freq.clone()
        self.attention_scale = 1.0
        self.yarn = None
        return None

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, start=0):
        T = q.shape[-2]
        t = torch.arange(start, start + T, device=q.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)                # (T, head_dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)              # (T, head_dim)
        # THE SCALE MULTIPLIES BOTH TABLES, so q and k each come back scaled and the logit they
        # form carries its square. At attention_scale 1.0 -- every training run, and every
        # evaluation at or below the training length -- this is the identity and the arithmetic
        # is bit-for-bit what it was.
        cos = emb.cos()[None, None].to(q.dtype)              # (1, 1, T, head_dim)
        sin = emb.sin()[None, None].to(q.dtype)
        if self.attention_scale != 1.0:
            cos = cos * self.attention_scale
            sin = sin * self.attention_scale
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, n_embd, max_len=8192):
        super().__init__()
        pe = torch.zeros(max_len, n_embd)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, n_embd, 2).float() * (-math.log(10000.0) / n_embd))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x, start=0):
        T = x.size(1)
        return x + self.pe[start:start + T].unsqueeze(0).to(x.dtype)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, n_embd, max_len=8192):
        super().__init__()
        self.emb = nn.Embedding(max_len, n_embd)

    def forward(self, x, start=0):
        T = x.size(1)
        pos = torch.arange(start, start + T, device=x.device)
        return x + self.emb(pos).unsqueeze(0)


def build_positional_encoding(kind, n_embd, max_len):
    kind = (kind or "sinusoidal").lower()
    if kind.startswith("sin"):
        return SinusoidalPositionalEncoding(n_embd, max_len)
    if kind.startswith("learn"):
        return LearnedPositionalEncoding(n_embd, max_len)
    raise ValueError(f"unknown positional encoding '{kind}' (sinusoidal|learned)")
