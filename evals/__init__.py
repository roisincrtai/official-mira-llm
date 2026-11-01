"""
evals -- measurements taken on a TRAINED model, separate from the stages that train it.

    python evals/eval.py --stages dpo                held-out evaluation of an aligned stage
    python evals/eval_pretrain_context_length.py     does a PRETRAINED model use a long context

A stage package (pretrain/, instruct_sft/, cot/, ...) owns an algorithm that produces a
checkpoint. Everything here owns a QUESTION asked of a checkpoint that already exists: nothing
in this package writes weights, and each module runs against any stage's output without the
pipeline being involved.

EITHER FORM WORKS -- `python evals/<name>.py` and `python -m evals.<name>`. Each module puts
the project root on sys.path itself, because the root is on the path when python is handed a
module and is not when it is handed a file, and `python evals/eval.py` is what a person types.
"""
