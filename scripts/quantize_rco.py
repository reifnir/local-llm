#!/usr/bin/env python3
"""Quantize a BF16 Qwen3.8-27B (or a fine-tune of it) to GGUF with ISTA-DASLab's RCO tensor allocation.

Runs on the GPU instance (terraform/ec2.tf), not in this repo's containers. DASLab publish, for each
of their GSQ-RCO GGUFs, the quantization type RCO picked for every tensor, plus the importance matrix
they used. This script applies exactly that allocation with stock llama-quantize, so the output
matches their files tensor for tensor and in size (IQ3_S: 3.50 bpw, 11.8 GB).

What it does NOT reproduce is GSQ itself, the learned per-weight rounding. DASLab's GGUF tooling is
unpublished, and the public GSQ code only refines K-quants, while most of these tensors are IQ types.
Quality therefore lands between a plain imatrix quant and theirs. --eval measures where: perplexity
and KL divergence against the BF16 model, and with --compare-daslab, of DASLab's own file as well.

Steps; each one is skipped when its output already exists, so a rerun resumes:
  1. build llama.cpp at the release the server uses, for this machine's GPU
  2. fetch the allocations and imatrix from DASLab's repo at one commit, checking every hash
  3. convert the safetensors to a BF16 GGUF, plus the vision projector
  4. quantize with one exact-name override per tensor, then check every tensor's type
  5. optionally, perplexity and KLD against BF16 on wikitext-2
  6. write SHA256SUMS and a model card next to the files, ready for `hf upload`

Needs git, cmake, gcc-c++, the CUDA toolkit (the Deep Learning AMI has it) and uv. Standard library
only, Python 3.9+ (the system python3 on Amazon Linux 2023 works); llama.cpp's own Python tools run
in a venv the script creates.

  python3 quantize_rco.py /opt/dlami/nvme/Qwen3.8-27B --eval --compare-daslab
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

LLAMA_CPP_TAG = "b12004"  # keep in step with the llama.cpp image in compose.yaml
LLAMA_CPP_GIT = "https://github.com/ggml-org/llama.cpp"
DASLAB_REPO = "ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF"
DASLAB_STEM = "Qwen3.8-27B-GSQ-RCO"
DASLAB_IMATRIX = "imatrix-qwen3.8-27b.gguf"
TYPES = ("IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S")  # DASLab's four sizes: 2.50 / 2.75 / 3.00 / 3.50 bpw
WIKITEXT_URL = "https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip"

HF = "https://huggingface.co"
CHUNK = 8 << 20

# Reads tensor names, types and the total element count from a GGUF; run with llama.cpp's gguf-py.
GGUF_INFO = """\
import json, sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
print(json.dumps({"types": {t.name: t.tensor_type.name for t in r.tensors},
                  "n_elements": sum(int(t.n_elements) for t in r.tensors)}))
"""


class Fail(Exception):
    """A user-facing error: printed without a traceback."""


def step(msg: str) -> None:
    print(f"\n== {msg}", flush=True)


def run(cmd: list, log: Path | None = None, cwd: Path | None = None, env: dict | None = None) -> None:
    """Run a command, echoing its output; with `log`, also keep the output there."""
    cmd = [str(c) for c in cmd]
    print("  $ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    if log is None:
        rc = subprocess.run(cmd, cwd=cwd, env=env).returncode
    else:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as f, subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                                  stderr=subprocess.STDOUT, text=True, errors="replace") as p:
            for line in p.stdout:
                f.write(line)
                sys.stdout.write(line)
        rc = p.returncode
    if rc:
        raise Fail(f"{Path(cmd[0]).name} failed (exit {rc})" + (f"; full output in {log}" if log else ""))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


# ── Hugging Face ────────────────────────────────────────────────────────────────


def hf_open(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": "local-llm/1.0"})
    token = os.environ.get("HF_TOKEN")
    if token:
        # unredirected: the token goes to huggingface.co only, never to the CDN it redirects to
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        raise Fail(f"HTTP {e.code} for {url}") from None


def hf_json(path: str) -> object:
    with hf_open(f"{HF}/api/{path}") as r:
        return json.load(r)


def daslab_tree(revision: str) -> dict[str, dict]:
    tree = hf_json(f"models/{DASLAB_REPO}/tree/{revision}?recursive=true")
    return {e["path"]: e for e in tree if e["type"] == "file"}


def fetch(tree: dict[str, dict], revision: str, path: str, dest: Path) -> Path:
    """Download one DASLab file and check it against the commit's hash: sha256 for LFS files,
    the git blob id for small ones. An existing copy is re-checked rather than trusted."""
    entry = tree.get(path)
    if entry is None:
        raise Fail(f"{DASLAB_REPO}@{revision[:12]} has no file {path}")
    if "lfs" in entry:
        expected, digest = entry["lfs"]["oid"], sha256_file
    else:
        expected = entry["oid"]
        digest = lambda p: hashlib.sha1(b"blob %d\0" % p.stat().st_size + p.read_bytes()).hexdigest()
    if dest.exists() and digest(dest) == expected:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    url = f"{HF}/{DASLAB_REPO}/resolve/{revision}/{urllib.parse.quote(path)}"
    with hf_open(url) as r, part.open("wb") as out:
        total, done, last = int(r.headers.get("Content-Length", 0)), 0, 0.0
        while chunk := r.read(CHUNK):
            out.write(chunk)
            done += len(chunk)
            if total > CHUNK and time.monotonic() - last > 1:
                last = time.monotonic()
                print(f"\r  {path}  {done / 1e9:6.2f}/{total / 1e9:.2f} GB", end="", flush=True)
    if total > CHUNK:
        print()
    if digest(part) != expected:
        part.unlink()
        raise Fail(f"hash mismatch for {DASLAB_REPO}/{path} at {revision[:12]}; download discarded")
    os.replace(part, dest)
    return dest


def parse_allocation(path: Path) -> dict[str, str]:
    """`<tensor name>: <ggml type>` lines; `#` lines are the header."""
    alloc = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            name, _, qtype = line.partition(":")
            alloc[name.strip()] = qtype.strip()
    if not alloc:
        raise Fail(f"no tensor assignments in {path}")
    return alloc


# ── llama.cpp ───────────────────────────────────────────────────────────────────


class LlamaCpp:
    def __init__(self, root: Path):
        self.root = root
        self.bin = root / "build" / "bin"
        self.python = root / ".venv" / "bin" / "python"

    def ensure(self) -> None:
        tools = ("llama-quantize", "llama-imatrix", "llama-perplexity")
        if all((self.bin / t).exists() for t in tools) and self.python.exists():
            return
        step(f"building llama.cpp {LLAMA_CPP_TAG}")
        missing = [c for c in ("git", "cmake", "g++", "uv", "nvidia-smi") if not shutil.which(c)]
        if missing:
            raise Fail(f"missing {', '.join(missing)}: sudo dnf install -y git cmake gcc-c++ "
                       f"(uv: curl -LsSf https://astral.sh/uv/install.sh | sh)")
        if not self.root.exists():
            run(["git", "clone", "--depth", "1", "--branch", LLAMA_CPP_TAG, LLAMA_CPP_GIT, self.root])
        env = dict(os.environ)
        if not shutil.which("nvcc") and Path("/usr/local/cuda/bin/nvcc").exists():
            env["PATH"] = "/usr/local/cuda/bin:" + env["PATH"]
        caps = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True).stdout.split()
        archs = ";".join(sorted({c.replace(".", "") for c in caps}))  # e.g. 12.0 -> 120 (Blackwell)
        run(["cmake", "-B", "build", "-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={archs}",
             "-DLLAMA_CURL=OFF", "-DCMAKE_BUILD_TYPE=Release"], cwd=self.root, env=env)
        run(["cmake", "--build", "build", "-j", os.cpu_count(), "--target", *tools], cwd=self.root, env=env)
        run(["uv", "venv", "--allow-existing", "-p", "3.12", ".venv"], cwd=self.root)
        run(["uv", "pip", "install", "--python", self.python, "-r",
             "requirements/requirements-convert_hf_to_gguf.txt"], cwd=self.root)

    def gguf_info(self, path: Path) -> dict:
        env = dict(os.environ, PYTHONPATH=str(self.root / "gguf-py"))
        out = subprocess.run([self.python, "-c", GGUF_INFO, path], capture_output=True, text=True, env=env)
        if out.returncode:
            raise Fail(f"could not read {path}:\n{out.stderr.strip()}")
        return json.loads(out.stdout.strip().splitlines()[-1])

    def convert(self, src: Path, dest: Path, extra: list[str], log: Path) -> None:
        """Convert into an empty scratch dir and move the single result into place, since --mmproj
        renames its output file."""
        tmp = dest.parent / f".convert-{dest.stem}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        run([self.python, self.root / "convert_hf_to_gguf.py", src, "--outtype", "bf16",
             "--outfile", tmp / dest.name, *extra], log=log)
        made = list(tmp.glob("*.gguf"))
        if len(made) != 1:
            raise Fail(f"expected one GGUF from the converter in {tmp}, found {len(made)}")
        os.replace(made[0], dest)
        tmp.rmdir()


# ── steps ───────────────────────────────────────────────────────────────────────


def check_tensor_set(alloc: dict[str, str], bf16_types: dict[str, str], alloc_name: str) -> None:
    missing, extra = sorted(set(alloc) - set(bf16_types)), sorted(set(bf16_types) - set(alloc))
    if missing or extra:
        raise Fail(f"the converted model's tensors don't match {alloc_name}:\n"
                   f"  in the allocation but not the model ({len(missing)}): {missing[:8]}\n"
                   f"  in the model but not the allocation ({len(extra)}): {extra[:8]}\n"
                   f"  The allocation only fits Qwen3.8-27B's architecture; a model with MTP tensors "
                   f"needs --mtp with.")


def quantize(llama: LlamaCpp, bf16: Path, out: Path, qtype: str, alloc: dict[str, str],
             imatrix: Path, logs: Path) -> dict[str, str]:
    """llama-quantize with every tensor pinned to its allocated type. Patterns are regexes searched
    anywhere in the name and the first match wins, so each one is anchored: a bare `blk.1.ffn_up`
    would also catch blk.10-19."""
    overrides = logs / f"{out.stem}.tensor-types.txt"
    overrides.parent.mkdir(parents=True, exist_ok=True)
    overrides.write_text("".join(f"^{re.escape(n)}$={t.lower()}\n" for n, t in alloc.items()))
    part = out.with_name(out.name + ".part")
    run([llama.bin / "llama-quantize", "--imatrix", imatrix, "--tensor-type-file", overrides,
         bf16, part, qtype, os.cpu_count()], log=logs / f"{out.stem}.quantize.log")
    got = llama.gguf_info(part)["types"]
    wrong = [(n, t, got.get(n)) for n, t in alloc.items() if got.get(n) != t]
    if wrong or len(got) != len(alloc):
        rejected = out.with_name(out.stem + ".rejected.gguf")
        os.replace(part, rejected)
        lines = "\n".join(f"    {n}: want {t}, got {g}" for n, t, g in wrong[:20])
        raise Fail(f"{len(wrong)} tensors in {out.name} don't have their allocated type "
                   f"(kept as {rejected.name}):\n{lines}")
    os.replace(part, out)
    return got


def eval_kld(llama: LlamaCpp, model: Path, base_kld: Path, log: Path) -> dict[str, float]:
    if not log.exists() or "Mean    KLD" not in log.read_text():
        run([llama.bin / "llama-perplexity", "-m", model, "--kl-divergence-base", base_kld,
             "--kl-divergence", "-c", "512", "-ngl", "99"], log=log)
    text = log.read_text()
    found = {}
    for key, pattern in (("ppl", r"Mean PPL\(Q\)\s*:\s*([\d.]+)"), ("kld", r"Mean\s+KLD:\s*([\d.]+)"),
                         ("top1", r"Same top p:\s*([\d.]+)")):
        if m := re.search(pattern, text):
            found[key] = float(m.group(1))
    return found


def model_card(args, rows: list[dict], revision: str, imatrix_note: str, eval_md: str) -> str:
    files = "\n".join(f"| `{r['file']}` | {r['bpw']:.2f} | {r['size'] / 1e9:.1f} GB | "
                      f"`{r['alloc']}` |" for r in rows)
    return f"""---
base_model: {args.base_model}
base_model_relation: quantized
license: {args.license}
library_name: gguf
tags:
  - gguf
  - imatrix
  - mixed-precision
---

# {args.name} GGUF (RCO allocation)

Mixed-precision GGUF quantizations of [{args.base_model}](https://huggingface.co/{args.base_model}).
Every tensor uses the quantization type that RCO assigned in
[{DASLAB_REPO}](https://huggingface.co/{DASLAB_REPO}/tree/{revision}) (commit `{revision[:12]}`),
so the layout and size match those files. Each tensor was quantized with stock `llama-quantize`
(llama.cpp {LLAMA_CPP_TAG}), **not** with GSQ's learned rounding, so quality is not expected to match
DASLab's GSQ-RCO files exactly. {imatrix_note}

| File | bpw | Size | Allocation |
|---|---|---|---|
{files}

The `mmproj` file (if present) is the vision projector at BF16. `tensor-allocation/` holds the
allocation each file was built from, and `SHA256SUMS` the file hashes.
{eval_md}
## Credits

The per-tensor allocation is DASLab's RCO search result. Please cite RCO
([arXiv:2605.00649](https://arxiv.org/abs/2605.00649)) and GSQ
([arXiv:2604.18556](https://arxiv.org/abs/2604.18556)) if you use it.
"""


# ── main ────────────────────────────────────────────────────────────────────────


def main() -> None:
    nvme = Path("/opt/dlami/nvme")
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("src", type=Path, help="Hugging Face-format BF16 model directory")
    p.add_argument("--name", help="output name prefix (default: the source directory's name)")
    p.add_argument("--work", type=Path, default=(nvme if nvme.is_dir() else Path.cwd()) / "quant",
                   help="working directory (default: /opt/dlami/nvme/quant, the local NVMe disk)")
    p.add_argument("--types", default="IQ3_S",
                   help=f"comma-separated DASLab sizes to build, or 'all' ({', '.join(TYPES)})")
    p.add_argument("--mtp", choices=("without", "with", "both"), default="without",
                   help="build without the MTP speculative-decoding head (default), with it, or both")
    p.add_argument("--calib", type=Path,
                   help="compute a fresh imatrix from this text file instead of using DASLab's. "
                        "Do this for a fine-tuned or abliterated model")
    p.add_argument("--chunks", type=int, default=1000,
                   help="imatrix chunks of 4096 tokens with --calib (default 1000, as DASLab used)")
    p.add_argument("--daslab-revision", help="DASLab repo commit to use (default: current main, printed)")
    p.add_argument("--no-mmproj", action="store_true", help="skip the vision projector")
    p.add_argument("--eval", action="store_true", help="perplexity + KL divergence vs BF16 on wikitext-2")
    p.add_argument("--eval-chunks", type=int, default=100,
                   help="512-token chunks for --eval (default 100; the BF16 logits file is ~12 GB)")
    p.add_argument("--compare-daslab", action="store_true",
                   help="with --eval, also download and score DASLab's own GSQ-RCO files")
    p.add_argument("--base-model", default="Qwen/Qwen3.8-27B", help="base_model in the model card")
    p.add_argument("--license", default="apache-2.0", help="license in the model card (the base model's)")
    p.add_argument("--llama-cpp", type=Path, help="llama.cpp checkout (default: <work>/llama.cpp)")
    args = p.parse_args()

    src = args.src.resolve()
    if not (src / "config.json").exists():
        raise Fail(f"{src} has no config.json: pass a Hugging Face-format model directory")
    args.name = args.name or src.name
    types = list(TYPES) if args.types == "all" else [t.strip().upper() for t in args.types.split(",")]
    if bad := [t for t in types if t not in TYPES]:
        raise Fail(f"unknown --types {bad}; DASLab published {', '.join(TYPES)}")
    variants = {"without": [""], "with": ["-mtp"], "both": ["", "-mtp"]}[args.mtp]
    if args.compare_daslab and not args.eval:
        raise Fail("--compare-daslab needs --eval")

    work = args.work.resolve()
    gguf_dir, logs = work / "gguf", work / "logs"
    out = work / "out" / f"{args.name}-GGUF"
    for d in (gguf_dir, logs, out / "tensor-allocation"):
        d.mkdir(parents=True, exist_ok=True)

    llama = LlamaCpp((args.llama_cpp or work / "llama.cpp").resolve())
    llama.ensure()

    step(f"DASLab artifacts ({DASLAB_REPO})")
    revision = args.daslab_revision or hf_json(f"models/{DASLAB_REPO}/revision/main")["sha"]
    if not args.daslab_revision:
        print(f"  using main = {revision}; pass --daslab-revision {revision} to pin it")
    tree = daslab_tree(revision)
    allocs = {}
    for v in variants:
        for t in types:
            rel = f"tensor-allocation/{DASLAB_STEM}-{t}{v}.rco-allocation.txt"
            allocs[t, v] = parse_allocation(fetch(tree, revision, rel, out / rel))

    for v in variants:
        bf16 = gguf_dir / f"{args.name}-BF16{v}.gguf"
        if not bf16.exists():
            step(f"converting to {bf16.name}")
            llama.convert(src, bf16, [] if v else ["--no-mtp"], logs / f"convert{v}.log")
    mmproj = out / f"mmproj-{args.name}-BF16.gguf"
    if not args.no_mmproj and not mmproj.exists():
        step("converting the vision projector")
        llama.convert(src, mmproj, ["--mmproj"], logs / "convert-mmproj.log")

    bf16_info = {v: llama.gguf_info(gguf_dir / f"{args.name}-BF16{v}.gguf") for v in variants}
    for (t, v), alloc in allocs.items():
        check_tensor_set(alloc, bf16_info[v]["types"], f"{DASLAB_STEM}-{t}{v}")

    if args.calib:
        imatrix = out / f"imatrix-{args.name}.gguf"
        imatrix_note = (f"The importance matrix was computed from this model "
                        f"({args.chunks} chunks of 4096 tokens) and is included.")
        if not imatrix.exists():
            step(f"computing the imatrix from {args.calib}")
            part = imatrix.with_name(imatrix.name + ".part")
            run([llama.bin / "llama-imatrix", "-m", gguf_dir / f"{args.name}-BF16{variants[0]}.gguf",
                 "-f", args.calib.resolve(), "-o", part, "-c", "4096", "--chunks", args.chunks, "-ngl", "99"],
                log=logs / "imatrix.log")
            os.replace(part, imatrix)
    else:
        imatrix = fetch(tree, revision, DASLAB_IMATRIX, out / DASLAB_IMATRIX)
        imatrix_note = f"The importance matrix is DASLab's (`{DASLAB_IMATRIX}`, included)."

    rows = []
    for (t, v), alloc in allocs.items():
        dest = out / f"{args.name}-RCO-{t}{v}.gguf"
        if dest.exists():
            print(f"\n== {dest.name} exists; checking it")
            got = llama.gguf_info(dest)["types"]
            if any(got.get(n) != q for n, q in alloc.items()):
                raise Fail(f"{dest} doesn't match its allocation; delete it and rerun")
        else:
            step(f"quantizing {dest.name}")
            quantize(llama, gguf_dir / f"{args.name}-BF16{v}.gguf", dest, t, alloc, imatrix, logs)
        size = dest.stat().st_size
        theirs = tree.get(f"{DASLAB_STEM}-{t}{v}.gguf", {}).get("size")
        rows.append({"type": t, "variant": v, "file": dest.name, "path": dest, "size": size,
                     "bpw": size * 8 / bf16_info[v]["n_elements"], "theirs": theirs,
                     "alloc": f"tensor-allocation/{DASLAB_STEM}-{t}{v}.rco-allocation.txt"})
        print(f"  {dest.name}: {size / 1e9:.2f} GB, {rows[-1]['bpw']:.2f} bpw"
              + (f" (DASLab's: {theirs / 1e9:.2f} GB)" if theirs else ""))

    eval_md = ""
    if args.eval:
        step("evaluating against BF16 on wikitext-2")
        eval_dir = work / "eval"
        wiki = eval_dir / "wiki.test.raw"
        if not wiki.exists():
            eval_dir.mkdir(parents=True, exist_ok=True)
            zpath = eval_dir / "wikitext-2-raw-v1.zip"
            with hf_open(WIKITEXT_URL) as r, zpath.open("wb") as f:
                shutil.copyfileobj(r, f)
            with zipfile.ZipFile(zpath) as z:
                wiki.write_bytes(z.read("wikitext-2-raw/wiki.test.raw"))
        base_kld = eval_dir / f"{args.name}-BF16.c512x{args.eval_chunks}.kld"
        base_log = logs / f"eval-bf16-c512x{args.eval_chunks}.log"
        if not base_kld.exists():
            part = base_kld.with_name(base_kld.name + ".part")
            run([llama.bin / "llama-perplexity", "-m", gguf_dir / f"{args.name}-BF16{variants[0]}.gguf",
                 "-f", wiki, "-c", "512", "--chunks", args.eval_chunks, "--kl-divergence-base", part,
                 "-ngl", "99"], log=base_log)
            os.replace(part, base_kld)
        m = re.search(r"Final estimate: PPL = ([\d.]+)", base_log.read_text()) if base_log.exists() else None
        table = [("BF16", m and float(m.group(1)), 0.0, 100.0)]
        for r in rows:
            res = eval_kld(llama, r["path"], base_kld, logs / f"eval-{r['path'].stem}.log")
            table.append((r["file"], res.get("ppl"), res.get("kld"), res.get("top1")))
            if args.compare_daslab and r["theirs"]:
                name = f"{DASLAB_STEM}-{r['type']}{r['variant']}.gguf"
                theirs = fetch(tree, revision, name, eval_dir / "daslab" / name)
                res = eval_kld(llama, theirs, base_kld, logs / f"eval-daslab-{Path(name).stem}.log")
                table.append((f"{name} (DASLab, GSQ-RCO)", res.get("ppl"), res.get("kld"), res.get("top1")))
        fmt = lambda x, spec: "?" if x is None else format(x, spec)
        eval_md = ("\n## Evaluation\n\nwikitext-2 test, 512-token context, "
                   f"{args.eval_chunks} chunks, against this model at BF16. Lower KLD is closer to BF16.\n\n"
                   "| Model | PPL | Mean KLD | Same top token |\n|---|---|---|---|\n"
                   + "".join(f"| {n} | {fmt(p, '.4f')} | {fmt(k, '.5f')} | {fmt(s, '.2f')}% |\n"
                             for n, p, k, s in table))
        print(eval_md)

    step("model card and checksums")
    (out / "README.md").write_text(model_card(args, rows, revision, imatrix_note, eval_md))
    sums = [f"{sha256_file(f)}  {f.relative_to(out)}\n" for f in sorted(out.rglob("*"))
            if f.is_file() and f.suffix in (".gguf", ".txt") and f.name != "SHA256SUMS"]
    (out / "SHA256SUMS").write_text("".join(sums))
    print(f"""
Done: {out}
Review README.md, then upload (repos are public by default; --private keeps it to you):
  uvx --from huggingface_hub==2.4.0 hf repos create <you>/{args.name}-GGUF --private
  uvx --from huggingface_hub==2.4.0 hf upload <you>/{args.name}-GGUF {out} .
The NVMe disk is wiped when the instance stops: upload or copy {out} first.""")


if __name__ == "__main__":
    try:
        main()
    except Fail as e:
        print(f"\nerror: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
