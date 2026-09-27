"""Post-hoc checks on the narrator's output.

Two failures, opposite in shape: `validator` catches an answer that says more
than the tools support; `completeness` catches one that says less.
"""
from backend.grounding.validator import (ID_PATTERNS, check, correction_prompt,
                                         corpus, normalize, supported_numbers)
from backend.grounding.completeness import complete as complete_answer

__all__ = ["check", "correction_prompt", "corpus", "normalize", "ID_PATTERNS",
           "supported_numbers", "complete_answer"]
