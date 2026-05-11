"""ReasoningTerminationTask — predict whether the model's CoT is about to end.

Task 1 of cot-proxy-tasks. Upstream historically used a multi-phase pipeline
that resampled 50 continuations per prefix to estimate P(`</think>` within
next 100 tokens). This module is a **simplified rebuild** that derives
per-rollout-prefix binary labels directly from the original rollout's
thinking text — no resampling, no judge model. The trade-off: we get a
binary label per rollout-prefix pair, not a calibrated probability.

Labeling logic: for a prefix at position p in thinking content of total
length L (in characters), label "yes" if (L - p) <= LOOKAHEAD_CHARS,
else "no". This approximates the upstream behavior of "is `</think>`
within ~100 tokens of this prefix" for English text.

Interface mirrors `HintedCotTask` / `MinMajAnswerTask`:
- ``__init__(subject_model, data_dir)``
- ``run_data(max_questions, num_rollouts)``: OpenRouter calls, saves
  per-question rollout JSONs.
- ``build_dataset(num_prefixes_per_rollout)``: derives binary labels and
  returns a `DataSlice` with a single train_df.
"""

from __future__ import annotations

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai
import pandas as pd
from tqdm import tqdm

from ..base import BaseTask
from ...data_slice import DataSlice

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_WORKERS = 30
LOOKAHEAD_CHARS = 400  # ~100 tokens for English

# Hardcoded prompts that elicit chain-of-thought reasoning.
# Small enough for local infra validation; expand for production.
DEFAULT_PROMPTS: List[Dict[str, str]] = [
    {"id": "math_001", "text": "What is 47 * 23? Show your reasoning."},
    {"id": "math_002", "text": "If a train leaves Boston at 3pm going 60mph and another leaves NYC at 4pm going 70mph, when do they meet? (Boston to NYC is ~200 miles.) Show your work."},
    {"id": "logic_001", "text": "Three friends split a $90 dinner bill equally. Then they realize they each have a $10 coupon. How much does each person actually pay?"},
    {"id": "logic_002", "text": "Alice is older than Bob. Bob is older than Charlie. Charlie is older than Dave. Who is the youngest? Explain your reasoning."},
    {"id": "puzzle_001", "text": "A bat and a ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?"},
]


class ReasoningTerminationTask(BaseTask):
    """Generate CoT rollouts and produce per-rollout-prefix binary labels."""

    def __init__(
        self,
        subject_model: str,
        data_dir: Optional[Path] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_workers: int = DEFAULT_MAX_WORKERS,
        api_key: Optional[str] = None,
    ):
        name = f"reasoning_termination-{subject_model.split('/')[-1]}"
        super().__init__(name, data_dir)

        self.subject_model = subject_model
        self.temperature = temperature
        self.max_workers = max_workers
        self.rollouts_dir = self.data_dir / "rollouts"

        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if self.api_key:
            self.client = openai.OpenAI(
                base_url=OPENROUTER_BASE_URL,
                api_key=self.api_key,
            )
        else:
            self.client = None

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def run_data(
        self,
        prompts: Optional[List[Dict[str, str]]] = None,
        max_questions: int = 5,
        num_rollouts: int = 5,
        verbose: bool = True,
    ) -> None:
        """Generate CoT rollouts via OpenRouter, one JSON per question.

        Saves to ``self.rollouts_dir/{qid}.json`` with shape:
        ``{question_id, question, runs: [{rollout_idx, thinking, answer}]}``.
        Per-question file presence makes re-runs idempotent.
        """
        if self.client is None:
            raise RuntimeError("No OpenRouter API key available.")

        if prompts is None:
            prompts = DEFAULT_PROMPTS
        prompts = prompts[:max_questions]
        if verbose:
            print(f"Loaded {len(prompts)} prompts")

        self.rollouts_dir.mkdir(parents=True, exist_ok=True)
        to_run = [p for p in prompts if not (self.rollouts_dir / f"{p['id']}.json").exists()]
        if verbose:
            print(f"{len(to_run)} to generate ({len(prompts) - len(to_run)} done)")

        for q in tqdm(to_run, desc="Questions", disable=not verbose):
            self._generate_question_rollouts(q, num_rollouts)

    def _generate_question_rollouts(self, q: Dict[str, str], num_rollouts: int) -> None:
        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {ex.submit(self._call_model, q["text"]): i for i in range(num_rollouts)}
            for fut in as_completed(futs):
                ridx = futs[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    print(f"  error for {q['id']} rollout {ridx}: {e}")
                    continue
                results.append({"rollout_idx": ridx, **r})
        results.sort(key=lambda x: x["rollout_idx"])

        with open(self.rollouts_dir / f"{q['id']}.json", "w") as f:
            json.dump({
                "question_id": q["id"],
                "question": q["text"],
                "runs": results,
            }, f, indent=2)

    def _call_model(self, prompt: str) -> Dict[str, str]:
        try:
            response = self.client.chat.completions.create(
                model=self.subject_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=self.temperature,
                extra_body={"reasoning": {"enabled": True}},
                timeout=90,
            )
            msg = response.choices[0].message
            thinking = ""
            extra = getattr(msg, "model_extra", {}) or {}
            if extra.get("reasoning"):
                thinking = extra["reasoning"]
            elif hasattr(msg, "reasoning_content") and msg.reasoning_content:
                thinking = msg.reasoning_content
            return {"thinking": thinking or "", "answer": msg.content or ""}
        except Exception:
            return {"thinking": "", "answer": ""}

    def get_data(self, load: bool = False):
        existed = self.rollouts_dir.exists() and any(self.rollouts_dir.glob("*.json"))
        if not load:
            return existed
        return existed or None

    def build_dataset(
        self,
        num_prefixes_per_rollout: int = 3,
        lookahead_chars: int = LOOKAHEAD_CHARS,
        seed: int = 42,
    ) -> DataSlice:
        """Sample prefix points per rollout and assign binary labels.

        For prefix at position ``p`` in thinking of length ``L``, the
        rollout's thinking will end within ``lookahead_chars`` characters
        iff ``(L - p) <= lookahead_chars``. That's the binary label.

        Returns a ``DataSlice`` with all rows in ``train_df`` (single split
        — produces one labeled corpus, no internal split).
        """
        rng = random.Random(seed)
        rows: List[Dict[str, Any]] = []

        for fp in sorted(self.rollouts_dir.glob("*.json")):
            data = json.loads(fp.read_text())
            qid = data.get("question_id", fp.stem)
            qtext = data.get("question", "")
            for run in data.get("runs", []):
                cot = run.get("thinking", "")
                if not cot or len(cot) < 50:
                    continue
                ridx = int(run.get("rollout_idx", 0))
                L = len(cot)

                # Sample prefix points uniformly over [10, L-1]
                positions = sorted(rng.sample(
                    range(10, L), min(num_prefixes_per_rollout, max(1, L - 10))
                ))
                for pidx, p in enumerate(positions):
                    prefix = cot[:p]
                    label = "yes" if (L - p) <= lookahead_chars else "no"
                    rows.append({
                        "question_id": qid,
                        "rollout_idx": ridx,
                        "prefix_idx": pidx,
                        "cot_prefix": prefix,
                        "label": label,
                        "question_text": qtext,
                        "cot_length": L,
                        "prefix_position": p,
                    })

        df = pd.DataFrame(rows)
        if df.empty:
            return DataSlice()
        return DataSlice(train_df=df, val_df=None, test_df=None)
