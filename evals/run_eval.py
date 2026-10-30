"""Command-line entry point for the SafeSpace evaluation suite.

Offline (router only, no model needed):
    uv run python evals/run_eval.py

Live benchmark against local Ollama models (needs `ollama serve` and the models pulled):
    uv run python evals/run_eval.py --live --models llama3.2 qwen2.5:3b --judge llama3.2

Reports are written to evals/reports/.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT)]

from evals.harness import live_benchmark, load_cases, offline_eval, summarize_benchmark  # noqa: E402

REPORTS = Path(__file__).with_name("reports")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="also benchmark the ReAct agent on Ollama models")
    parser.add_argument("--models", nargs="+", default=["llama3.2"])
    parser.add_argument("--judge", help="Ollama model used as LLM-as-judge for answer quality")
    args = parser.parse_args()

    cases = load_cases()
    REPORTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    offline = offline_eval(cases)
    print(f"Router on {offline['cases']} golden cases")
    print(f"  routing accuracy     {offline['routing_accuracy']:.1%}")
    print(f"  risk exact accuracy  {offline['risk_exact_accuracy']:.1%}")
    for level, m in offline["risk"].items():
        print(f"  {level:<6} precision {m['precision']:.2f}  recall {m['recall']:.2f}")
    crisis = offline["crisis_detection"]
    print(f"  any-crisis recall    {crisis['recall']:.2f}  (missed {crisis['fn']})")
    for name, split in offline["splits"].items():
        print(f"  [{name:<7}] {split['cases']:>2} cases  routing {split['routing_accuracy']:.1%}  "
              f"crisis recall {split['crisis_detection']['recall']:.2f}")
    for f in offline["failures"]:
        print(f"  ✗ {f['id']:<11} route {f['route']} (want {f['expected_route']}), "
              f"risk {f['risk']} (want {f['expected_risk']})")
    (REPORTS / f"offline-{stamp}.json").write_text(json.dumps(offline, indent=2))

    if args.live:
        from ai_agent import build_react_agent
        from langchain_ollama import ChatOllama

        judge = ChatOllama(model=args.judge, temperature=0) if args.judge else None
        rows = live_benchmark(cases, args.models, build_react_agent, judge_llm=judge)
        table = summarize_benchmark(rows)
        print("\nReAct agent benchmark\n" + table.to_string())
        (REPORTS / f"live-{stamp}.json").write_text(json.dumps(rows, indent=2))
        table.to_csv(REPORTS / f"live-{stamp}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
