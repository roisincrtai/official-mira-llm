import numpy as np
from pretrain.epoch import EpochCorpus

class Stream:
    def __init__(self, start, count):
        self.values=np.arange(start,start+count,dtype=np.int64)
        self.n_tokens=count;self.path=str(start);self.sig=str(count);self.eos=0
    def __len__(self):return 1
    def doc(self,i):return self.values.tolist()
    def read(self,start,n):return self.values[start:start+n]
class Corpus:
    def __init__(self,lengths):
        self.streams=[Stream(1000*i+1,n) for i,n in enumerate(lengths)]
        self.n_tokens=sum(lengths)
    def __len__(self):return len(self.streams)
    def doc(self,i):return self.streams[i].doc(0)

def consume(corpus,windows,batch,regions,micro,start=0,stop=None):
    readers=[EpochCorpus(corpus,windows,batch,regions=regions,start=start) for _ in range(regions)]
    found=[]
    for first,last,context,local,_ in readers[0].training_plan:
        for step in range(max(start,first), min(last,stop if stop is not None else last)):
            for rank,reader in enumerate(readers):
                for offset in range(0,local,micro or local):
                    size=min(micro or local,local-offset)
                    ids,attn,mask=reader.batch(size,context-1,regions=regions,region=rank if regions>1 else None)
                    for row in range(size):
                        for col in range(1,context):
                            if mask[row,col]:found.append((int(ids[row,col-1]),int(ids[row,col])))
    return found,readers[0].steps
def test_coverage():
    cases=0
    for lengths in [[2],[3],[21],[27,17],[1,9,1,53],[104,77,203]]:
        corpus=Corpus(lengths)
        wanted=[(int(a),int(b)) for s in corpus.streams for a,b in zip(s.values[:-1],s.values[1:])]
        for windows in [[4],[4,8],[8,16,32]]:
            for regions in [1,2,4]:
                for batch in [1,3]:
                    for micro in [0,1,2]:
                        got,steps=consume(corpus,windows,batch,regions,micro)
                        assert sorted(got)==sorted(wanted),(lengths,windows,batch,regions,micro)
                        assert len(got)==len(set(got))
                        cut=steps//2
                        before,_=consume(corpus,windows,batch,regions,micro,stop=cut)
                        after,_=consume(corpus,windows,batch,regions,micro,start=cut)
                        assert before+after==got
                        cases+=1
    print(f'PASS: {cases} exact-coverage and resume cases, including partial batches, multiple corpora, context schedules and 1/2/4 data ranks.')

def test_training_resume():
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace
    import torch
    from helpers import common, load_ckpt
    from model import lm
    from model.mira_llm import MiraLLM

    with tempfile.TemporaryDirectory(prefix="mira-epoch-test-") as directory:
        root = Path(directory)
        yaml = root / "config.yaml"
        yaml.write_text("mira_config: mira-xs\ncontext_window: 4,8\nbatch: 2\npretrain_epochs: 1\n")
        args = common.parse_args(["--config", str(yaml)], stage="pretrain")
        args.max_len = 8
        args.resume = True
        args.mixed_precision = False
        args.distributed_parallelism = "ddp"
        args.loss_chunk = 0
        args.micro_batch = 0
        args.batch = 2
        args.context_window = "4,8"
        args.checkpoint_every_steps = 1
        args.plot_every_steps = 0
        corpus = Corpus([11, 16])
        for stream in corpus.streams:
            stream.values = np.arange(1, stream.n_tokens + 1, dtype=np.int64)

        def model():
            return MiraLLM(vocab_size=64, n_layer=1, n_embd=16, block_size=8,
                           lattice_d_e=16, coefficient_bottleneck=4,
                           memory_dimension=2, lattice_order=2, ffn_hidden=32)

        torch.manual_seed(13)
        initial = model().state_dict()

        def run(path, stop=None):
            instance = model()
            instance.load_state_dict(initial)
            checkpoint = load_ckpt(str(path), "pretrain")
            data = EpochCorpus(corpus, [4, 8], 2,
                               start=checkpoint["step"] if checkpoint else 0)
            assert common.corpus_prompts(data, tok=None) == []
            lm.train(instance, SimpleNamespace(device="cpu"), data, str(path), args,
                     lambda message: None, stage="pretrain", steps=data.steps,
                     lr=args.peak_lr, stop_at=stop, profile=False,
                     checkpoint_extra=data.metadata)
            return instance

        full = run(root / "full")
        run(root / "resumed", stop=1)
        resumed = run(root / "resumed")
        for key, value in full.state_dict().items():
            torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
            assert torch.isfinite(value).all()
        print("PASS: real MIRA forward/backward and bit-identical interrupted/resumed training.")


def test_pretraining_configuration():
    import contextlib
    import io
    import tempfile
    from pathlib import Path
    from helpers import common
    from model import lm
    with tempfile.TemporaryDirectory(prefix="mira-config-test-") as directory:
        path = Path(directory) / "config.yaml"
        path.write_text("mira_config: mira-xs\ncontext_window: 4,8\nbatch: 2\n")
        with contextlib.redirect_stdout(io.StringIO()):
            args = common.parse_args(["--config", str(path)], stage="pretrain")
        args.max_len = 8
        assert args.pretrain_epochs == 1
        plan = lm.context_plan_for(args, 10, Corpus([20]))
        assert all(row[-1] == 0 for row in plan)
        for option in (["--micro_batch", "1"], ["--pretrain_epochs", "0"]):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    common.parse_args(["--config", str(path)] + option, stage="pretrain")
                except SystemExit as error:
                    assert error.code == 2
                else:
                    raise AssertionError("Pretraining accepted a forbidden setting")
    print("PASS: one epoch is enforced and pretraining micro-batching is disabled.")


if __name__ == "__main__":
    test_coverage()
    test_training_resume()
    test_pretraining_configuration()
