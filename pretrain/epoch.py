import bisect
import math


class EpochCorpus:
    def __init__(self, corpus, windows, batch, regions=1, start=0):
        self.corpus = corpus
        self.streams = list(getattr(corpus, "streams", [corpus]))
        self.n_tokens = corpus.n_tokens
        self.n_docs = len(corpus)
        self.regions = regions
        if not windows or min(windows) < 2 or batch < 1 or regions < 1:
            raise ValueError("One epoch requires context >= 2 and positive batch and region counts")
        counts = [max(0, s.n_tokens - 1) for s in self.streams]
        total = sum(counts)
        if not total:
            raise ValueError("One epoch requires at least one next-token target")
        self.training_plan = []
        self.segments = []
        self.step = start
        self.offset = 0
        prefix = [0]
        for count in counts:
            prefix.append(prefix[-1] + count)
        at = 0
        for index, context in enumerate(windows):
            lo = total * index // len(windows)
            hi = total * (index + 1) // len(windows)
            local_batch = max(1, round(batch * max(windows) / context))
            pieces, ends, rows = [], [], 0
            for stream_index, count in enumerate(counts):
                first = max(lo, prefix[stream_index])
                last = min(hi, prefix[stream_index + 1])
                if last <= first:
                    continue
                length = last - first
                pieces.append((stream_index, first - prefix[stream_index], length, rows))
                rows += math.ceil(length / (context - 1))
                ends.append(rows)
            if not rows:
                continue
            steps = math.ceil(rows / (local_batch * regions))
            self.training_plan.append((at, at + steps, context, local_batch, 0))
            self.segments.append((pieces, ends, rows))
            at += steps
        self.steps = at
        if not 0 <= start <= self.steps:
            raise ValueError("Checkpoint step lies outside the one-epoch plan")
        self.metadata = {
            "pretrain_epochs": 1,
            "epoch_streams": [[s.path, s.sig, s.n_tokens] for s in self.streams],
            "epoch_plan": [list(row) for row in self.training_plan],
            "epoch_regions": regions,
        }

    def __len__(self):
        return len(self.corpus)

    def doc(self, index):
        return self.corpus.doc(index)

    def batch(self, batch, seq_len, generator=None, device=None, regions=1, region=None):
        import numpy as np
        import torch
        if regions != self.regions:
            raise ValueError("Data parallelism changed after the one-epoch plan was created")
        index = next((i for i, row in enumerate(self.training_plan)
                      if row[0] <= self.step < row[1]), None)
        if index is None:
            raise ValueError("The complete epoch has already been consumed")
        first_step, _, context, local_batch, _ = self.training_plan[index]
        if seq_len != context - 1:
            raise ValueError("Context does not match the one-epoch plan")
        process_batch = local_batch if region is not None else local_batch * regions
        if self.offset + batch > process_batch:
            raise ValueError("Micro-batch exceeds the current epoch step")
        pieces, ends, rows = self.segments[index]
        first_row = ((self.step - first_step) * local_batch * regions
                     + (int(region) * local_batch if region is not None else 0) + self.offset)
        ids = np.full((batch, context), self.streams[0].eos, dtype=np.int64)
        valid = np.zeros((batch, context), dtype=np.float32)
        for row in range(batch):
            position = first_row + row
            if position >= rows:
                continue
            piece = bisect.bisect_right(ends, position)
            stream_index, start, length, row_base = pieces[piece]
            offset = (position - row_base) * (context - 1)
            targets = min(context - 1, length - offset)
            ids[row, :targets + 1] = self.streams[stream_index].read(start + offset, targets + 1)
            valid[row, :targets + 1] = 1
        self.offset += batch
        if self.offset == process_batch:
            self.step += 1
            self.offset = 0
        return (torch.as_tensor(ids, device=device), torch.as_tensor(valid, device=device),
                torch.as_tensor(valid, device=device))
