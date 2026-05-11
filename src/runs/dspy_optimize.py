"""
DSPy prompt optimization for the zero-shot LLM-monitor baseline.

Uses a small subset of the `train` split as DSPy bootstrap pool, the `val`
split as the optimization dev set, and evaluates the compiled program on
`test` and `ood_test`. Per-task config (CoT field, label mapping, task
description, optional context field) is shared with `zero_shot_monitor.py`.

Install:
    pip install -U dspy

Run:
    OPENAI_API_KEY=sk-... python -m src.runs.dspy_optimize \\
        --task 4 --train-n 100 --dev-n 100 --eval-n 100

Outputs (under results/dspy/{task}/):
    program.json                 compiled DSPy program (signature + demos)
    metrics.json                 baseline + optimized metrics on val/test/ood_test
    predictions_{split}.jsonl    per-sample preds for each eval split
"""

from __future__ import annotations

import argparse
import json
import random
import resource
import sys
from pathlib import Path


def _raise_fd_limit() -> None:
    """Raise the soft file-descriptor limit toward the hard limit.

    macOS defaults to 256 which DSPy + LiteLLM blow through quickly when
    bootstrapping in parallel ([Errno 24] Too many open files).
    """
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(hard, 65536) if hard != resource.RLIM_INFINITY else 65536
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            print(f"[fd-limit] raised RLIMIT_NOFILE soft from {soft} to {target}")
    except Exception as e:  # noqa: BLE001
        print(f"[fd-limit] could not raise RLIMIT_NOFILE: {e}")


_raise_fd_limit()

import dspy  # noqa: E402  (import after fd-limit bump is intentional)

# Reuse task configs from the zero-shot script.
from src.runs.zero_shot_monitor import TASKS, TaskConfig, truncate_cot

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"
RESULTS = ROOT / "results" / "dspy"


# ---------- DSPy signature & program -----------------------------------------

class MonitorSignature(dspy.Signature):
    """Classify a property of a language model's chain of thought (CoT).

    The CoT may be a partial reasoning trace cut off mid-thought - do not
    continue or complete it; only classify it. Output a single integer
    label (0 or 1) according to the per-task instructions.
    """

    task_description: str = dspy.InputField(
        desc="What property of the CoT to classify, with definitions of label 0 vs label 1."
    )
    context: str = dspy.InputField(
        desc="Optional surrounding context (e.g. the prompt the model is reasoning about). May be empty."
    )
    cot: str = dspy.InputField(desc="The chain of thought to classify.")
    rationale: str = dspy.OutputField(desc="One short sentence explaining the decision.")
    label: int = dspy.OutputField(desc="Predicted label: 0 or 1.")


class MonitorProgram(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classify = dspy.Predict(MonitorSignature)

    def forward(self, task_description: str, context: str, cot: str):
        return self.classify(task_description=task_description, context=context, cot=cot)


# ---------- data loading -----------------------------------------------------

def task_description_for(cfg: TaskConfig) -> str:
    return (
        f"{cfg.description}\n\n"
        f"LABELS:\n"
        f"- 0 = {cfg.neg_class_desc}\n"
        f"- 1 = {cfg.pos_class_desc}"
    )


def to_example(cfg: TaskConfig, sample: dict, max_cot_chars: int) -> dspy.Example | None:
    raw = sample.get("label")
    if raw == cfg.pos_label:
        gold = 1
    elif raw == cfg.neg_label:
        gold = 0
    else:
        return None
    cot = truncate_cot(sample.get(cfg.cot_field) or "", max_cot_chars)
    context = ""
    if cfg.context_field and sample.get(cfg.context_field):
        context = str(sample[cfg.context_field])[: max_cot_chars // 2]
    return dspy.Example(
        task_description=task_description_for(cfg),
        context=context,
        cot=cot,
        label=gold,
    ).with_inputs("task_description", "context", "cot")


def load_split(
    cfg: TaskConfig,
    split: str,
    n: int | None,
    seed: int,
    max_cot_chars: int,
    balance: bool = True,
) -> list[dspy.Example]:
    folder = DATASETS / str(cfg.task_id) / "qwen-3-32b" / split
    if not folder.exists():
        raise SystemExit(f"missing folder: {folder}")
    files = sorted(folder.glob("*.json"))
    rng = random.Random(seed)
    rng.shuffle(files)

    pos: list[dspy.Example] = []
    neg: list[dspy.Example] = []
    for f in files:
        with open(f) as fh:
            ex = to_example(cfg, json.load(fh), max_cot_chars)
        if ex is None:
            continue
        (pos if ex.label == 1 else neg).append(ex)
        if not balance and n is not None and len(pos) + len(neg) >= n:
            break
        if balance and n is not None and len(pos) >= n // 2 and len(neg) >= n // 2:
            break

    if balance and n is not None:
        per = n // 2
        out = pos[:per] + neg[:per]
    else:
        out = (pos + neg)[: n if n is not None else None]
    rng.shuffle(out)
    return out


# ---------- metric & evaluation ----------------------------------------------

def accuracy_metric(example, pred, trace=None) -> bool:  # noqa: ARG001
    _ = trace
    try:
        return int(pred.label) == int(example.label)
    except Exception:
        return False


def evaluate(program: MonitorProgram, examples: list[dspy.Example]) -> dict:
    rows: list[dict] = []
    tp = fp = tn = fn = err = 0
    for ex in examples:
        try:
            out = program(
                task_description=ex.task_description,
                context=ex.context,
                cot=ex.cot,
            )
            pred = int(out.label)
            rationale = getattr(out, "rationale", "")
        except Exception as e:  # noqa: BLE001
            pred, rationale = -1, f"<error: {e!s}>"
            err += 1
        gold = int(ex.label)
        if pred == 1 and gold == 1: tp += 1
        elif pred == 0 and gold == 1: fn += 1
        elif pred == 0 and gold == 0: tn += 1
        elif pred == 1 and gold == 0: fp += 1
        rows.append({"gold": gold, "pred": pred, "rationale": rationale})

    n = tp + tn + fp + fn
    pos, neg = tp + fn, tn + fp
    tpr = tp / pos if pos else 0.0
    tnr = tn / neg if neg else 0.0
    return {
        "rows": rows,
        "metrics": {
            "n": n, "errors": err,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": (tp + tn) / n if n else 0.0,
            "balanced_accuracy": (tpr + tnr) / 2,
            "tpr": tpr, "tnr": tnr,
            "gmean2": tpr * tnr,
        },
    }


# ---------- main -------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True, choices=sorted(TASKS))
    ap.add_argument("--model", default="openai/gpt-5.4-mini",
                    help="DSPy LM string, e.g. 'openai/gpt-5.4-mini' or 'openai/gpt-4o-mini'.")
    ap.add_argument("--prompt-model", default=None,
                    help="Optional separate LM for proposing instructions (MIPRO).")
    ap.add_argument("--optimizer", default="bootstrap",
                    choices=["bootstrap", "mipro"],
                    help="bootstrap = BootstrapFewShotWithRandomSearch; mipro = MIPROv2.")
    ap.add_argument("--mipro-auto", default=None,
                    choices=[None, "light", "medium", "heavy"],
                    help="MIPROv2 budget preset. If set, --num-candidates and --num-trials are ignored.")
    ap.add_argument("--num-trials", type=int, default=None,
                    help="MIPROv2 only. Optimization trials (used when --mipro-auto is unset). "
                         "Defaults to 2 * num_candidates.")
    ap.add_argument("--train-n", type=int, default=80,
                    help="Bootstrap pool size (drawn from train split).")
    ap.add_argument("--dev-n", type=int, default=100,
                    help="Optimization dev-set size (drawn from val split).")
    ap.add_argument("--eval-n", type=int, default=100,
                    help="Eval set size for test and ood_test.")
    ap.add_argument("--max-bootstrapped-demos", type=int, default=4)
    ap.add_argument("--max-labeled-demos", type=int, default=8)
    ap.add_argument("--num-candidates", type=int, default=6,
                    help="Random-search trials (bootstrap) or candidate programs (MIPRO).")
    ap.add_argument("--num-threads", type=int, default=2,
                    help="Concurrency inside DSPy. Keep low on macOS (default 2) to avoid "
                         "running out of file descriptors. Bump after `ulimit -n 65536`.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-cot-chars", type=int, default=12_000)
    ap.add_argument("--out-dir", default=str(RESULTS))
    args = ap.parse_args()

    cfg = TASKS[args.task]
    out_dir = Path(args.out_dir) / str(args.task)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Configure DSPy LM.
    lm = dspy.LM(args.model, temperature=0.0, max_tokens=512)
    dspy.configure(lm=lm)

    # Load splits.
    print(f"[task {args.task}] loading splits...")
    train = load_split(cfg, "train", args.train_n, args.seed, args.max_cot_chars)
    dev = load_split(cfg, "val", args.dev_n, args.seed + 1, args.max_cot_chars)
    test = load_split(cfg, "test", args.eval_n, args.seed + 2, args.max_cot_chars)
    ood = load_split(cfg, "ood_test", args.eval_n, args.seed + 3, args.max_cot_chars)
    print(f"  train={len(train)} dev={len(dev)} test={len(test)} ood_test={len(ood)}")

    program = MonitorProgram()

    # Baseline (no optimization).
    print("[baseline] evaluating un-optimized program on dev...")
    baseline_dev = evaluate(program, dev)
    print(f"  baseline dev: {baseline_dev['metrics']}")

    # Optimize.
    if args.optimizer == "bootstrap":
        from dspy.teleprompt import BootstrapFewShotWithRandomSearch
        optimizer = BootstrapFewShotWithRandomSearch(
            metric=accuracy_metric,
            max_bootstrapped_demos=args.max_bootstrapped_demos,
            max_labeled_demos=args.max_labeled_demos,
            num_candidate_programs=args.num_candidates,
            num_threads=args.num_threads,
        )
        print(f"[optimize:bootstrap] compiling with {len(train)} train / {len(dev)} dev...")
        compiled = optimizer.compile(program, trainset=train, valset=dev)
    else:
        from dspy.teleprompt import MIPROv2
        prompt_lm = dspy.LM(args.prompt_model, temperature=0.7, max_tokens=1024) if args.prompt_model else lm
        # MIPROv2 in DSPy >= 3.x: `auto` and (num_candidates, num_trials) are mutually exclusive.
        # If user passed --mipro-auto, defer to the budget preset; otherwise use explicit candidates/trials.
        # In DSPy 3.x, `num_candidates` is an __init__ arg but `num_trials` is a compile() arg.
        # `auto` and (num_candidates, num_trials) are mutually exclusive.
        init_kwargs: dict = {
            "metric": accuracy_metric,
            "prompt_model": prompt_lm,
            "task_model": lm,
            "init_temperature": 0.7,
            "verbose": True,
            "auto": args.mipro_auto,
        }
        compile_kwargs: dict = {
            "trainset": train,
            "valset": dev,
            "max_bootstrapped_demos": args.max_bootstrapped_demos,
            "max_labeled_demos": args.max_labeled_demos,
        }
        if args.mipro_auto is None:
            num_trials = args.num_trials if args.num_trials is not None else max(args.num_candidates * 2, 5)
            init_kwargs["num_candidates"] = args.num_candidates
            compile_kwargs["num_trials"] = num_trials

        optimizer = MIPROv2(**init_kwargs)
        print(
            f"[optimize:mipro] compiling with {len(train)} train / {len(dev)} dev "
            f"(auto={args.mipro_auto}, num_candidates={init_kwargs.get('num_candidates')}, "
            f"num_trials={compile_kwargs.get('num_trials')})..."
        )
        compiled = optimizer.compile(program, **compile_kwargs)

    # Evaluate.
    print("[eval] running compiled program on dev/test/ood_test...")
    results = {
        "baseline_dev": baseline_dev["metrics"],
        "compiled_dev": evaluate(compiled, dev)["metrics"],
        "compiled_test": None,
        "compiled_ood_test": None,
    }
    test_eval = evaluate(compiled, test)
    ood_eval = evaluate(compiled, ood)
    results["compiled_test"] = test_eval["metrics"]
    results["compiled_ood_test"] = ood_eval["metrics"]

    # Persist.
    compiled.save(str(out_dir / "program.json"))
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(
            {"task": args.task, "model": args.model, "optimizer": args.optimizer, **results},
            fh, indent=2,
        )
    for name, ev in (("test", test_eval), ("ood_test", ood_eval), ("dev", evaluate(compiled, dev))):
        with open(out_dir / f"predictions_{name}.jsonl", "w") as fh:
            for r in ev["rows"]:
                fh.write(json.dumps(r) + "\n")

    print(f"\nwrote {out_dir}/program.json")
    print(f"wrote {out_dir}/metrics.json")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
