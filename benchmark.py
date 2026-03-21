"""
Benchmark script for LISF Agent: Haiku vs Sonnet comparison.

Sends 3 messy real-user-style prompts to the running server, measures
TTFT / total time / tool calls, checks for path leaks, and writes
results to subagents_outputs/benchmark_results.md.

Usage:
    # Start server with desired model, then:
    python benchmark.py --model haiku
    python benchmark.py --model sonnet
    python benchmark.py --model sonnet --clean   # overwrite previous results
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Test prompts -- designed to NOT match FAQ cache, force tool use, and test
# article hints + path-leak prevention.
# ---------------------------------------------------------------------------

PROMPTS = [
    # 1. Vague + typos -- tests disambiguation and general knowledge
    "oye q onda con las reaseguradoras extranjeras, como les afecta la lisf? se pueden registrar o q pex",
    # 2. Specific article + informal -- tests article hints + path leak prevention
    "me puedes explicar el articulo 216 y el 217 porfa, es que no le entiendo a lo de reservas tecnicas",
    # 3. Complex multi-part -- tests depth, tool use, and response completeness
    "cuales son las diferencias entre una institucion de seguros y una sociedad mutualista, y q pasa si operan sin autorizacion?",
]

SHORT_LABELS = [
    "reaseguradoras extranjeras...",
    "articulo 216 y 217...",
    "diferencias seguros vs mutualista...",
]

# Leak detection: match internal paths/tool names but avoid false positives
# on normal Spanish prose ("leer" != "Read", etc.)
LEAK_RE = re.compile(r"docs/|lisf_md|\.md\b|(?<!\w)Read(?!\w)|(?<!\w)Grep(?!\w)")

RESULTS_PATH = Path(__file__).parent / "subagents_outputs" / "benchmark_results.md"


def send_prompt(url: str, message: str) -> dict:
    """Send a prompt to /api/chat and consume the SSE stream.

    Returns dict with keys: ttft, total, text, tool_count, leaks.
    """
    text_chunks: list[str] = []
    tool_count = 0
    ttft: float | None = None

    t0 = time.monotonic()

    resp = requests.post(
        f"{url}/api/chat",
        json={"message": message, "history": []},
        stream=True,
        timeout=180,
    )
    resp.raise_for_status()

    try:
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            raw = line[len("data:"):].strip()
            if not raw:
                continue
            try:
                evt = json.loads(raw)
            except json.JSONDecodeError:
                continue

            etype = evt.get("type")
            if etype == "text":
                if ttft is None:
                    ttft = time.monotonic() - t0
                text_chunks.append(evt.get("content", ""))
            elif etype == "tool_use":
                tool_count += 1
            elif etype == "done":
                break
            elif etype == "error":
                text_chunks.append(f"[ERROR: {evt.get('content', '')}]")
                break
    finally:
        resp.close()

    total = time.monotonic() - t0
    full_text = "".join(text_chunks)
    leaks = LEAK_RE.findall(full_text)

    return {
        "ttft": ttft if ttft is not None else total,
        "total": total,
        "text": full_text,
        "tool_count": tool_count,
        "leaks": leaks,
    }


def format_results(model: str, results: list[dict]) -> str:
    """Format a single model run as markdown."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"## Run: {model} -- {now}",
        "",
        "| # | Prompt | TTFT (s) | Total (s) | Tools | Leaks | Length |",
        "|---|--------|----------|-----------|-------|-------|--------|",
    ]

    ttfts, totals, tools = [], [], []
    for i, (res, label) in enumerate(zip(results, SHORT_LABELS), 1):
        leak_str = "NONE" if not res["leaks"] else ", ".join(set(res["leaks"]))
        lines.append(
            f"| {i} | {label} | {res['ttft']:.1f} | {res['total']:.1f} "
            f"| {res['tool_count']} | {leak_str} | {len(res['text']):,} |"
        )
        ttfts.append(res["ttft"])
        totals.append(res["total"])
        tools.append(res["tool_count"])

    avg_ttft = sum(ttfts) / len(ttfts)
    avg_total = sum(totals) / len(totals)
    avg_tools = sum(tools) / len(tools)
    lines.append("")
    lines.append(
        f"**Averages:** TTFT {avg_ttft:.2f}s | Total {avg_total:.2f}s | Tools {avg_tools:.1f}"
    )
    lines.append("")

    # Full responses
    lines.append("### Responses")
    lines.append("")
    for i, (res, label) in enumerate(zip(results, SHORT_LABELS), 1):
        lines.append(f"#### Prompt {i}: {label}")
        lines.append("")
        # Indent response as blockquote for readability
        for resp_line in res["text"].splitlines():
            lines.append(f"> {resp_line}")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="LISF Agent benchmark")
    parser.add_argument(
        "--model",
        required=True,
        help="Label for this run (e.g. 'haiku', 'sonnet'). The actual model is set on the server.",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the running LISF Agent (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Overwrite previous results instead of appending",
    )
    args = parser.parse_args()

    # Verify server is reachable
    try:
        requests.get(f"{args.url}/api/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"Cannot reach server at {args.url}: {e}")
        print("Start the server first, then run the benchmark.")
        return

    print(f"Running benchmark against {args.url} (label: {args.model})")
    print(f"Sending {len(PROMPTS)} prompts...\n")

    results = []
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"  [{i}/{len(PROMPTS)}] {SHORT_LABELS[i-1]}")
        res = send_prompt(args.url, prompt)
        leak_flag = " ** LEAKS DETECTED **" if res["leaks"] else ""
        print(
            f"           TTFT={res['ttft']:.1f}s  Total={res['total']:.1f}s  "
            f"Tools={res['tool_count']}  Len={len(res['text']):,}{leak_flag}"
        )
        results.append(res)

    # Build report
    report_section = format_results(args.model, results)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.clean or not RESULTS_PATH.exists():
        content = "# Benchmark: Haiku vs Sonnet\n\n" + report_section + "\n"
    else:
        content = RESULTS_PATH.read_text(encoding="utf-8")
        content += "\n---\n\n" + report_section + "\n"

    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"\nResults written to {RESULTS_PATH}")

    # Summary
    any_leaks = any(r["leaks"] for r in results)
    if any_leaks:
        print("\n** WARNING: Path leaks detected in responses! Check the report. **")
    else:
        print("\nNo path leaks detected -- regression test passed.")


if __name__ == "__main__":
    main()
