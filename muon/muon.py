"""Muon, with fp32 master weights, behind the same interface as helpers.utils.MasterAdamW.

    from muon import MasterMuon

MUON UPDATES A MATRIX BY ITS ORIENTATION, NOT ITS MAGNITUDE. AdamW rescales each weight by its
own gradient history, which treats a matrix as a bag of unrelated scalars. Muon takes the
momentum buffer of a 2D parameter and replaces it with the nearest semi-orthogonal matrix, so
every direction in the update is taken at the same step length and no singular direction can
dominate because it happened to have a large gradient. The update is the sign of the matrix in
the same sense that Adam's is the sign of a scalar.

THE ORTHOGONALIZATION IS A QUINTIC NEWTON-SCHULZ ITERATION and never an SVD. Five steps of

    X <- a X + (b A + c A^2) X,     A = X X^T

with the coefficients below drive the singular values of a Frobenius-normalized matrix towards
one. It is all matrix multiplication, it runs in bf16 without trouble, and it costs a handful of
small GEMMs per matrix per step -- against an SVD, which would be a different order of expense
and would not be worth the exactness.

NOT EVERY PARAMETER IS A MATRIX, and the ones that are not go to AdamW. Embeddings and the tied
output head are lookup tables whose rows are independent, biases and normalization gains are
vectors, and orthogonalizing any of them is meaningless. So this is a HYBRID: Muon for 2D
weights, AdamW for everything else, one `.step()` covering both. That split is Muon's own
convention and not an accommodation.

THE LEARNING RATE IS THE CALLER'S, UNCHANGED. Muon is normally run at a different rate from
AdamW; this class scales nothing and reads `lr` exactly as given.

fp32 masters are kept for the same reason MasterAdamW keeps them: the live parameters may be
bf16 under autocast, and a bf16 weight cannot absorb an update smaller than its own ulp.
"""
import torch


@torch.no_grad()
def orthogonalize(matrix, steps=5, eps=1e-7):
    """The nearest semi-orthogonal matrix to `matrix`, by quintic Newton-Schulz.

    The coefficients are tuned so the iteration converges from any Frobenius-normalized start
    rather than only from one already near orthogonal, which is what lets five steps do.

    THE SHORT SIDE IS PUT FIRST. The iteration forms X X^T, so working with the transpose when
    there are more rows than columns makes that product the smaller of the two and the whole
    iteration cheaper by the aspect ratio.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(int(steps)):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return (x.T if transposed else x).to(matrix.dtype)


class MasterMuon:
    """Muon for 2D weights, AdamW for the rest, fp32 masters throughout.

    Interface-compatible with helpers.utils.MasterAdamW: `param_groups` is the AdamW group list
    so a scheduler's lr edits reach both halves, and `state_dict`/`load_state_dict` round-trip.
    """

    def __init__(self, params, lr=1e-3, weight_decay=0.0, momentum=0.95,
                 nesterov=True, ns_steps=5):
        # A MODULE OR AN ITERABLE OF PARAMETERS. Given the module, the embedding is found and
        # excluded; given bare parameters there is no way to find it and the caller has said,
        # by what it passed, that it does not want it found.
        embeddings = set()
        if isinstance(params, torch.nn.Module):
            module = params
            for sub in module.modules():
                if isinstance(sub, torch.nn.Embedding) and sub.weight is not None:
                    embeddings.add(id(sub.weight))
            params = module.parameters()
        params = [p for p in params if p.requires_grad]
        # A PARAMETER IS A MATRIX OR IT IS NOT, and nothing else about it matters here -- except
        # the embedding, which is 2D and is not a matrix in the sense that matters. Its rows are
        # independent lookups, not directions of one linear map, so orthogonalizing across them
        # mixes unrelated tokens. THE TIED HEAD IS THE SAME TENSOR and so is excluded with it by
        # identity, which is the only test that stays correct while the tying does.
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self.ns_steps = int(ns_steps)
        self._muon = []                       # [(live, master, buffer)]
        adam_live = []
        for p in params:
            if p.ndim == 2 and min(p.shape) > 1 and id(p) not in embeddings:
                master = p.detach().clone().float()
                self._muon.append((p, master, torch.zeros_like(master)))
            else:
                adam_live.append(p)
        self._adam_pairs = [(p, p.detach().clone().float()) for p in adam_live]
        self.inner = torch.optim.AdamW(
            [m for _, m in self._adam_pairs] or [torch.zeros(1, requires_grad=True)],
            lr=lr, weight_decay=weight_decay)
        self._lr = float(lr)
        self.weight_decay = float(weight_decay)

    @property
    def param_groups(self):
        # THE SCHEDULER EDITS THIS AND BOTH HALVES MUST FOLLOW. AdamW's group is the one a
        # CosineLR will write `lr` into, so Muon reads its rate back out of the same place
        # rather than keeping a copy that would silently stay at the initial value.
        return self.inner.param_groups

    def zero_grad(self, set_to_none=True):
        for p, _, _ in self._muon:
            p.grad = None
        for p, _ in self._adam_pairs:
            p.grad = None

    @torch.no_grad()
    def step(self):
        lr = float(self.param_groups[0]["lr"])
        for live, master, buffer in self._muon:
            if live.grad is None:
                continue
            grad = live.grad.detach().float()
            buffer.mul_(self.momentum).add_(grad)
            direction = grad.add(buffer, alpha=self.momentum) if self.nesterov else buffer
            update = orthogonalize(direction, self.ns_steps)
            # THE ASPECT-RATIO SCALE. An orthogonalized update has unit singular values, so its
            # Frobenius norm is sqrt(min(rows, cols)) regardless of how large the matrix is;
            # this factor makes the step the same size per unit of fan-in whatever the shape.
            update = update * max(1.0, live.shape[0] / live.shape[1]) ** 0.5
            if self.weight_decay:
                master.mul_(1.0 - lr * self.weight_decay)
            master.add_(update, alpha=-lr)
            live.copy_(master.to(live.dtype))
        for p, m in self._adam_pairs:
            m.grad = None if p.grad is None else p.grad.detach().float()
        if self._adam_pairs:
            self.inner.step()
            for p, m in self._adam_pairs:
                p.copy_(m.to(p.dtype))

    def state_dict(self):
        return {"inner": self.inner.state_dict(),
                "muon_masters": [m for _, m, _ in self._muon],
                "muon_buffers": [b for _, _, b in self._muon],
                "adam_masters": [m for _, m in self._adam_pairs]}

    @torch.no_grad()
    def load_state_dict(self, sd):
        if not (isinstance(sd, dict) and "muon_masters" in sd):
            # A CHECKPOINT FROM THE OTHER OPTIMIZER. Its weights are already loaded into the
            # live parameters by the caller, so adopt those as masters and start the momentum
            # from zero rather than refusing the resume over optimizer state.
            for _, m, b in self._muon:
                b.zero_()
            for live, m, _ in self._muon:
                m.copy_(live.float())
            for p, m in self._adam_pairs:
                m.copy_(p.float())
            return
        self.inner.load_state_dict(sd["inner"])
        for (_, m, b), sm, sb in zip(self._muon, sd["muon_masters"], sd["muon_buffers"]):
            m.copy_(sm.to(m.device))
            b.copy_(sb.to(b.device))
        for (_, m), s in zip(self._adam_pairs, sd["adam_masters"]):
            m.copy_(s.to(m.device))
        for live, m, _ in self._muon:
            live.copy_(m.to(live.dtype))
        for p, m in self._adam_pairs:
            p.copy_(m.to(p.dtype))

    def describe(self):
        matrices = sum(m.numel() for _, m, _ in self._muon)
        others = sum(m.numel() for _, m in self._adam_pairs)
        total = matrices + others or 1
        return (f"Muon on {len(self._muon)} matrices ({matrices:,} parameters, "
                f"{100 * matrices / total:.0f}%), AdamW on {len(self._adam_pairs)} other "
                f"tensors ({others:,}); momentum {self.momentum:g}, "
                f"{self.ns_steps} Newton-Schulz steps")
