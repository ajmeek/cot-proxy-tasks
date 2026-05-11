"""AtypicalCotLengthTask — classify CoT as long/short by per-prompt z-score.

Task 7 of cot-proxy-tasks. Upstream's pipeline used vLLM at scale (200+
rollouts per prompt for stable stats) plus a bin-based greedy sampler for
class balance. This module is a **simplified rebuild**: per-prompt mean
and SD of CoT length (in characters) are computed from however many
rollouts we have, then each rollout is labeled "long" if z>+1, "short"
if z<-1, and dropped if |z|<=1.

Interface mirrors `HintedCotTask` / `MinMajAnswerTask`:
- ``__init__(subject_model, data_dir)``
- ``run_data(max_questions, num_rollouts)``: OpenRouter calls, saves
  per-question rollout JSONs.
- ``build_dataset()``: derives binary labels from per-prompt distribution
  stats and returns a ``DataSlice`` with all rows in train_df.
"""

from __future__ import annotations

import json
import os
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai
import pandas as pd
from tqdm import tqdm

from ..base import BaseTask
from ...data_slice import DataSlice

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TEMPERATURE = 0.9  # higher temp encourages length variance
DEFAULT_MAX_WORKERS = 30
SD_THRESHOLD = 1.0  # |z| > SD_THRESHOLD → labeled long/short
MIN_ROLLOUTS_FOR_STATS = 5  # need at least N rollouts to compute stable μ/σ

# Hardcoded prompts that elicit chain-of-thought reasoning of varying lengths.
# Same shape as task 1's; expand for production.
DEFAULT_PROMPTS: List[Dict[str, str]] = [
    {"id": "math_001", "text": "What is 47 * 23? Show your reasoning."},
    {"id": "math_002", "text": "If a train leaves Boston at 3pm going 60mph and another leaves NYC at 4pm going 70mph, when do they meet? (Boston to NYC is ~200 miles.) Show your work."},
    {"id": "puzzle_001", "text": "A bat and a ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?"},
    {"id": "logic_001", "text": "Three friends split a $90 dinner bill equally. Then they realize they each have a $10 coupon. How much does each person actually pay?"},
    {"id": "puzzle_002", "text": "A farmer needs to take a fox, a chicken, and a sack of grain across a river. The boat is small and can only carry the farmer plus one item. The fox would eat the chicken if left alone with it; the chicken would eat the grain. How does the farmer get all three across?"},
]


class AtypicalCotLengthTask(BaseTask):
    """Generate CoT rollouts and produce per-rollout long/short labels via z-score."""

    def __init__(
        self,
        subject_model: str,
        data_dir: Optional[Path] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_workers: int = DEFAULT_MAX_WORKERS,
        api_key: Optional[str] = None,
    ):
        name = f"atypical_cot_length-{subject_model.split('/')[-1]}"
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
        num_rollouts: int = 30,
        verbose: bool = True,
    ) -> None:
        """Generate CoT rollouts, one JSON per question with multiple runs."""
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
        sd_threshold: float = SD_THRESHOLD,
        min_rollouts_for_stats: int = MIN_ROLLOUTS_FOR_STATS,
    ) -> DataSlice:
        """Compute per-prompt μ/σ of CoT length, label each rollout's z-score.

        Drops:
        - Rollouts with empty thinking
        - Whole prompts with fewer than ``min_rollouts_for_stats`` valid rollouts
        - Rollouts with |z| <= ``sd_threshold`` (the ambiguous middle band)

        Returns ``DataSlice`` with all surviving rows in ``train_df``.
        """
        rows: List[Dict[str, Any]] = []

        for fp in sorted(self.rollouts_dir.glob("*.json")):
            data = json.loads(fp.read_text())
            qid = data.get("question_id", fp.stem)
            qtext = data.get("question", "")
            valid_runs = [
                r for r in data.get("runs", [])
                if r.get("thinking") and len(r["thinking"]) >= 20
            ]
            if len(valid_runs) < min_rollouts_for_stats:
                continue

            lengths = [len(r["thinking"]) for r in valid_runs]
            mu = statistics.mean(lengths)
            sigma = statistics.stdev(lengths) if len(lengths) >= 2 else 0.0
            if sigma <= 0:
                continue

            for run in valid_runs:
                cot = run["thinking"]
                length = len(cot)
                z = (length - mu) / sigma
                if abs(z) <= sd_threshold:
                    continue  # ambiguous middle, skip
                label = "long" if z > 0 else "short"
                rows.append({
                    "question_id": qid,
                    "rollout_idx": int(run.get("rollout_idx", 0)),
                    "chain_of_thought": cot,
                    "label": label,
                    "question_text": qtext,
                    "cot_length": length,
                    "z_score": float(z),
                    "prompt_mean_length": float(mu),
                    "prompt_std_length": float(sigma),
                })

        df = pd.DataFrame(rows)
        if df.empty:
            return DataSlice()
        return DataSlice(train_df=df, val_df=None, test_df=None)
