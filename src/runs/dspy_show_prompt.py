"""
Inspect a compiled DSPy program: dump the optimized instructions, the
selected few-shot demos, and one fully-rendered prompt as it would be
sent to the LM.

Usage:
    python -m src.runs.dspy_show_prompt --task 4
    python -m src.runs.dspy_show_prompt --task 4 --program results/dspy/4/program.json --n 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import dspy

from src.runs.dspy_optimize import MonitorProgram, TASKS, load_split

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True, choices=sorted(TASKS))
    ap.add_argument("--program", default=None,
                    help="Path to compiled program.json (defaults to results/dspy/{task}/program.json).")
    ap.add_argument("--model", default="openai/gpt-5.4-mini")
    ap.add_argument("--split", default="test", choices=["test", "ood_test", "val"])
    ap.add_argument("--n", type=int, default=1, help="How many sample prompts to render.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-cot-chars", type=int, default=12_000)
    args = ap.parse_args()

    program_path = Path(args.program or (ROOT / "results" / "dspy" / str(args.task) / "program.json"))
    if not program_path.exists():
        raise SystemExit(f"missing compiled program: {program_path}")

    raw = json.loads(program_path.read_text())
    print(f"=== Compiled program: {program_path} ===\n")

    # 1) Optimized instructions — what the model is told the task is.
    sig = raw.get("classify", {}).get("signature", {}) or raw.get("signature", {})
    instructions = sig.get("instructions") or "(none stored)"
    print("--- INSTRUCTIONS (signature.instructions) ---")
    print(instructions)
    print()

    # 2) Selected few-shot demos — what got bootstrapped onto the prompt.
    demos = raw.get("classify", {}).get("demos") or raw.get("demos") or []
    print(f"--- DEMOS ({len(demos)}) ---")
    for i, d in enumerate(demos):
        print(f"[demo {i}]")
        for k in ("task_description", "context", "cot", "rationale", "label"):
            if k in d:
                v = d[k]
                v_str = (v[:300] + "...[truncated]") if isinstance(v, str) and len(v) > 300 else v
                print(f"  {k}: {v_str}")
        print()

    # 3) Fully-rendered prompt for live samples.
    print(f"--- RENDERED PROMPTS (n={args.n}, split={args.split}) ---")
    dspy.configure(lm=dspy.LM(args.model, temperature=0.0, max_tokens=512))
    prog = MonitorProgram()
    prog.load(str(program_path))
    cfg = TASKS[args.task]
    examples = load_split(cfg, args.split, args.n, args.seed, args.max_cot_chars)
    for i, ex in enumerate(examples):
        print(f"\n----- sample {i} (gold={ex.label}) -----")
        prog(task_description=ex.task_description, context=ex.context, cot=ex.cot)
        dspy.inspect_history(n=1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
