"""Evaluation entry point.

    python eval.py kernel        deterministic layer vs the shipped answer keys
    python eval.py kernel -v     every individual check
    python eval.py llm           all 38 questions through the full chat stack
    python eval.py llm --tier 1  one tier only
    python eval.py custom        unseen questions: paraphrases, mess, out-of-scope
    python eval.py validate      check the shipped answer keys against the dataset
    python eval.py all           all three

`kernel` needs no API key and no network. It is the safety net: if the model
layer is unavailable, this still demonstrates that the system computes the
right answers.
"""
from __future__ import annotations

import sys


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "kernel"
    if mode in ("-h", "--help"):
        print(__doc__)
        return 0
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if mode == "kernel":
        from evals.kernel import main as run
        return run()
    if mode == "llm":
        from evals.llm import main as run
        return run()
    if mode == "custom":
        from evals.custom import main as run
        return run()
    if mode == "validate":
        from evals.validate_questions import main as run
        return run()
    if mode == "all":
        from evals.kernel import main as run_kernel
        rc = run_kernel()
        print()
        from evals.llm import main as run_llm
        rc = run_llm() or rc
        print()
        from evals.custom import main as run_custom
        return run_custom() or rc

    print(f"unknown mode {mode!r}\n")
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
