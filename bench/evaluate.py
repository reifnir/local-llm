"""Run HumanEval completions against their unit tests. Executes model-written code, so it only
runs in the `sandbox` container: no network, read-only filesystem, capped memory/CPU/processes.

usage: python evaluate.py /runs/<run-id>/completions.jsonl  >  eval.jsonl
"""

import json
import os
import resource
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

TIMEOUT_S = 20


def limits() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (TIMEOUT_S, TIMEOUT_S))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))


def program(rec: dict) -> str:
    imports = "\n".join(line for line in rec["prompt"].splitlines() if line.startswith(("import ", "from ")))
    code = rec["code"]
    if f"def {rec['entry_point']}" not in code:  # model returned only the body: continue the prompt
        code = rec["prompt"] + code
    return f"{imports}\n\n{code}\n\n{rec['test']}\n\ncheck({rec['entry_point']})\n"


def run(rec: dict) -> dict:
    result = {"task_id": rec["task_id"]}
    if rec.get("error"):
        return {**result, "passed": False, "reason": f"no completion: {rec['error']}"}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "solution.py")
        with open(path, "w") as f:
            f.write(program(rec))
        try:
            p = subprocess.run([sys.executable, "-I", "-B", path], cwd=d, capture_output=True, text=True,
                               timeout=TIMEOUT_S, preexec_fn=limits)
        except subprocess.TimeoutExpired:
            return {**result, "passed": False, "reason": "timeout"}
    if p.returncode == 0:
        return {**result, "passed": True}
    err = (p.stderr.strip().splitlines() or ["exit " + str(p.returncode)])[-1]
    return {**result, "passed": False, "reason": err[:200]}


def main() -> None:
    with open(sys.argv[1]) as f:
        records = [json.loads(line) for line in f if line.strip()]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for res in pool.map(run, records):
            print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
