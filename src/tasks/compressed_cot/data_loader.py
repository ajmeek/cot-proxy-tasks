"""
Data loader for compressed CoT task.

Originally re-exported five helpers from ``forced_response.data_loader``
(``get_latest_verification_dir`` etc.). Those have since been removed
upstream; the imports were dead. Replaced with stubs that raise
``NotImplementedError`` if called — the verification-rollouts pipeline
isn't exercised in our regen path (we feed source rollouts directly via
``CompressedCotTask.get_choice_distribution``).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...utils.questions import Question


def _stub(name: str):
    def fn(*_args: Any, **_kwargs: Any):
        raise NotImplementedError(
            f"{name}() is a verification-rollouts helper that was removed upstream. "
            f"This stub keeps imports working; provide source rollouts directly instead."
        )
    return fn


get_latest_verification_dir = _stub("get_latest_verification_dir")
get_verified_questions = _stub("get_verified_questions")
load_question_and_cot = _stub("load_question_and_cot")
load_verification_summary = _stub("load_verification_summary")
question_from_summary = _stub("question_from_summary")


__all__ = [
    "load_question_and_cot",
    "get_verified_questions",
    "get_latest_verification_dir",
    "load_verification_summary",
    "question_from_summary",
]
