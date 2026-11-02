#!/usr/bin/env python3
"""Benchmark the served LLM: speed (prompt processing, generation) and HumanEval completions.

Runs in the `tool` container. Model-written code is NOT executed here; bench/evaluate.py runs it
in the offline `sandbox` container, and `record` folds those results into bench/results.*.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import statistics
import sys
import time
import tomllib
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from models import ROOT, SERVER, Fail, get_model, is_installed, load_config, read_active

BENCH_DIR = ROOT / "bench"
RUNS_DIR = BENCH_DIR / "runs"
RESULTS_JSONL = BENCH_DIR / "results.jsonl"
RESULTS_MD = BENCH_DIR / "results.md"
DATASET = ROOT / "models" / "datasets" / "HumanEval.jsonl.gz"

PROMPT = ("Complete this Python function. Reply with the full implementation, including the signature "
          "and any imports it needs, in a single ```python code block.\n\n```python\n{prompt}```")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def settings() -> dict:
    return tomllib.loads((BENCH_DIR / "bench.toml").read_text())


def post(path: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(f"{SERVER}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def get(path: str) -> dict:
    with urllib.request.urlopen(f"{SERVER}{path}", timeout=10) as r:
        return json.load(r)


# ── dataset ─────────────────────────────────────────────────────────────────────


def load_dataset(cfg: dict) -> list[dict]:
    expected = cfg["dataset"]["sha256"]
    if not DATASET.exists():
        log(f"downloading {cfg['dataset']['name']} ...")
        DATASET.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(cfg["dataset"]["url"], timeout=60) as r:
            data = r.read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise Fail("HumanEval download does not match its pinned sha256; discarded")
        DATASET.write_bytes(data)
    data = DATASET.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise Fail(f"{DATASET.relative_to(ROOT)} does not match its pinned sha256: delete it to re-download")
    return [json.loads(line) for line in gzip.decompress(data).decode().splitlines() if line.strip()]


def pick(problems: list[dict], limit: str | int) -> list[dict]:
    if str(limit) == "all" or int(limit) >= len(problems):
        return problems
    n = int(limit)
    return [problems[i * len(problems) // n] for i in range(n)]  # evenly spread, deterministic


# ── speed ───────────────────────────────────────────────────────────────────────


def prompt_of_tokens(problems: list[dict], target: int) -> str:
    """Real code text, at least `target` tokens long (HumanEval prompts, repeated as needed)."""
    parts, total, i = [], 0, 0
    while total < target:
        p = problems[i % len(problems)]["prompt"]
        parts.append(p)
        total += len(post("/tokenize", {"content": p}, 30)["tokens"])
        i += 1
    return "\n".join(parts)


def measure_speed(cfg: dict, problems: list[dict], n_ctx: int) -> dict:
    out: dict = {}
    for target in cfg["speed"]["prompt_tokens"]:
        if target + 512 > n_ctx:
            log(f"  prompt {target // 1024}K: skipped (context is {n_ctx})")
            continue
        r = post("/completion", {"prompt": prompt_of_tokens(problems, target), "n_predict": 1,
                                 "cache_prompt": False}, 900)
        t = r["timings"]
        out[f"pp_{target // 1024}k"] = round(t["prompt_per_second"], 1)
        log(f"  prompt {t['prompt_n']} tokens: {t['prompt_per_second']:.0f} tok/s")
    rates = []
    for _ in range(cfg["speed"]["generate_runs"]):
        r = post("/completion", {"prompt": "def quicksort(arr):", "n_predict": cfg["speed"]["generate_tokens"],
                                 "ignore_eos": True, "cache_prompt": False}, 900)
        rates.append(r["timings"]["predicted_per_second"])
    out["tg"] = round(statistics.median(rates), 1)
    log(f"  generation: {out['tg']} tok/s (median of {len(rates)})")
    return out


# ── HumanEval ───────────────────────────────────────────────────────────────────


def extract_code(text: str, entry_point: str) -> str:
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.DOTALL)
    for block in reversed(blocks):
        if f"def {entry_point}" in block:
            return block
    return blocks[-1] if blocks else text


def run_humaneval(cfg: dict, problems: list[dict], model_id: str, out_path: Path) -> None:
    he = cfg["humaneval"]
    with out_path.open("w") as out:
        for n, p in enumerate(problems, 1):
            rec = {k: p[k] for k in ("task_id", "prompt", "entry_point", "test")}
            t0 = time.monotonic()
            try:
                r = post("/v1/chat/completions", {
                    "model": model_id, "seed": he["seed"], "max_tokens": he["max_tokens"],
                    "messages": [{"role": "user", "content": PROMPT.format(prompt=p["prompt"])}],
                }, he["request_timeout_s"])
                choice = r["choices"][0]
                rec |= {"code": extract_code(choice["message"].get("content") or "", p["entry_point"]),
                        "finish": choice.get("finish_reason"),
                        "tokens": r.get("usage", {}).get("completion_tokens", 0)}
            except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
                rec |= {"code": "", "error": str(e)[:200], "tokens": 0}
            rec["seconds"] = round(time.monotonic() - t0, 1)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            note = " (hit max_tokens)" if rec.get("finish") == "length" else rec.get("error", "")
            log(f"  [{n:3}/{len(problems)}] {p['task_id']:15} {rec['tokens']:6} tokens {rec['seconds']:7.1f}s {note}")


# ── commands ────────────────────────────────────────────────────────────────────


def cmd_active(args) -> None:
    active = (read_active() or {}).get("id")
    if not active:
        raise Fail("no model selected: pass MODEL=<id> or ALL=1")
    print(active)


def cmd_installed(args) -> None:
    _, models = load_config()
    print(" ".join(m["id"] for m in models.values() if m["kind"] == "llm" and is_installed(m)))


def cmd_run(args) -> None:
    cfg = settings()
    _, models = load_config()
    m = get_model(models, args.model)
    problems = load_dataset(cfg)
    props = get("/props")
    n_ctx = int(props.get("default_generation_settings", {}).get("n_ctx", 0))
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{m['id']}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True)
    log(f"speed ({m['id']}, context {n_ctx}):")
    speed = measure_speed(cfg, problems, n_ctx)
    chosen = pick(problems, args.limit or cfg["humaneval"]["limit"])
    log(f"HumanEval: {len(chosen)} problems")
    t0 = time.monotonic()
    run_humaneval(cfg, chosen, m["id"], run_dir / "completions.jsonl")
    meta = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "model": m["id"], "gguf": m.get("gguf"), "placement": m.get("placement"), "mtp": bool(m.get("mtp")),
        "n_ctx": n_ctx, "llama_cpp": props.get("build_info", "?"),
        "load_s": round(args.load_ms / 1000, 1) if args.load_ms is not None else None,
        "vram_mib": args.vram_mib, "humaneval_wall_s": round(time.monotonic() - t0),
        "dataset_sha256": cfg["dataset"]["sha256"], "seed": cfg["humaneval"]["seed"], **speed,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(run_id)


def cmd_record(args) -> None:
    run_dir = RUNS_DIR / args.run_id
    meta = json.loads((run_dir / "meta.json").read_text())
    completions = [json.loads(line) for line in (run_dir / "completions.jsonl").read_text().splitlines()]
    evals = {r["task_id"]: r for r in map(json.loads, (run_dir / "eval.jsonl").read_text().splitlines())}
    if len(evals) != len(completions):
        raise Fail(f"{args.run_id}: {len(evals)} results for {len(completions)} completions; sandbox run incomplete")
    passed = sum(evals[c["task_id"]]["passed"] for c in completions)
    row = meta | {
        "run_id": args.run_id,
        "humaneval_n": len(completions), "humaneval_passed": passed,
        "pass_at_1": round(passed / len(completions), 3),
        "mean_tokens": round(statistics.mean(c["tokens"] for c in completions)),
        "mean_s": round(statistics.mean(c["seconds"] for c in completions), 1),
        "truncated": sum(c.get("finish") == "length" for c in completions),
    }
    with RESULTS_JSONL.open("a") as f:
        f.write(json.dumps(row) + "\n")
    failures = [f"{t}: {r.get('reason', '')}" for t, r in evals.items() if not r["passed"]]
    (run_dir / "failures.txt").write_text("\n".join(failures) + "\n")
    log(f"{meta['model']}: HumanEval {passed}/{len(completions)} = {row['pass_at_1']:.1%}, "
        f"generation {meta['tg']} tok/s (failures: bench/runs/{args.run_id}/failures.txt)")
    write_report()


def fmt(v, spec: str = "", none: str = "-") -> str:
    return none if v is None else format(v, spec)


def write_report() -> None:
    rows = [json.loads(line) for line in RESULTS_JSONL.read_text().splitlines() if line.strip()]
    latest: dict[str, dict] = {}
    for r in rows:  # the newest run per model and HumanEval size
        latest[(r["model"], r["humaneval_n"])] = r
    table = sorted(latest.values(), key=lambda r: (-r["pass_at_1"], -r["tg"]))
    lines = [
        "# Benchmark results",
        "",
        "Generated by `make bench` from [results.jsonl](results.jsonl) (every run is kept there); "
        "newest run per model and problem count. Speed is single-user (one request at a time).",
        "",
        "| Model | Placement | HumanEval pass@1 | Tokens / problem | s / problem | Gen tok/s "
        "| Prompt 4K tok/s | Prompt 24K tok/s | Load s | VRAM GB | Context | Date | llama.cpp |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table:
        vram = fmt(r.get("vram_mib") and r["vram_mib"] / 1024, ".1f")
        mtp = " +MTP" if r.get("mtp") else ""
        cut = f", {r['truncated']} cut off" if r.get("truncated") else ""
        lines.append(
            f"| {r['model']} | {r['placement']}{mtp} "
            f"| {r['pass_at_1']:.1%} ({r['humaneval_passed']}/{r['humaneval_n']}){cut} "
            f"| {r['mean_tokens']} | {r['mean_s']} | {r['tg']} | {fmt(r.get('pp_4k'), '.0f')} "
            f"| {fmt(r.get('pp_24k'), '.0f')} | {fmt(r.get('load_s'))} | {vram} | {r['n_ctx']} "
            f"| {r['date'][:10]} | {r['llama_cpp'].split('-')[0]} |")
    lines += [
        "",
        "- **HumanEval pass@1**: share of problems whose code passes the hidden unit tests on the first "
        "try (model's own sampling settings, fixed seed). Small samples are noisy: at 40 problems, "
        "±8 points is within chance. Compare models at the same problem count.",
        "- **Tokens / problem** includes thinking tokens, so it shows how much a model reasons "
        "before answering (ThinkingCap's selling point).",
        "- **Prompt 24K** approximates reading Claude Code's first request.",
        "",
    ]
    RESULTS_MD.write_text("\n".join(lines))


def cmd_report(args) -> None:
    if not RESULTS_JSONL.exists():
        raise Fail("no results yet: run `make bench`")
    write_report()
    log(f"wrote {RESULTS_MD.relative_to(ROOT)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("active").set_defaults(fn=cmd_active)
    sub.add_parser("installed").set_defaults(fn=cmd_installed)
    sp = sub.add_parser("run")
    sp.add_argument("--model", required=True)
    sp.add_argument("--limit")
    sp.add_argument("--load-ms", type=int)
    sp.add_argument("--vram-mib", type=int)
    sp.set_defaults(fn=cmd_run)
    sp = sub.add_parser("record")
    sp.add_argument("run_id")
    sp.set_defaults(fn=cmd_record)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    args = p.parse_args()
    try:
        args.fn(args)
    except Fail as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
