#!/usr/bin/env bash
# Uploads quantize_rco.py's IQ3_S and IQ3_XXS output to a private Hugging Face repo, then checks the
# repo's hashes against SHA256SUMS. Safe to re-run: the upload resumes where it stopped.
#
#   export HF_TOKEN=<write token>
#   upload_gguf.sh <bf16-dir> [work-dir]
#
# bf16-dir and work-dir are the source directory and --work given to quantize_rco.py.
# The repo is reifnir/<bf16-dir's name>-GGUF. If it already exists, its visibility is left alone.
set -euo pipefail

usage="usage: $(basename "$0") <bf16-dir> [work-dir]"
fail() { echo "error: $*" >&2; exit 1; }
hf() { uvx --from huggingface_hub==2.4.0 hf "$@"; }

[ $# -ge 1 ] && [ $# -le 2 ] || fail "$usage"
[ -f "$1/config.json" ] || fail "$1 has no config.json: pass the BF16 model directory"
[ -n "${HF_TOKEN:-}" ] || fail "HF_TOKEN is not set: export a write token"
command -v uvx >/dev/null || fail "uvx not found: install uv and put ~/.local/bin on PATH"

name=$(basename "$(realpath "$1")")
if [ $# -eq 2 ]; then
  work=$2
elif [ -d /opt/dlami/nvme ]; then
  work=/opt/dlami/nvme/quant
else
  work=$PWD/quant
fi
out=$work/out/$name-GGUF
repo=reifnir/$name-GGUF

[ -d "$out" ] || fail "$out not found: run quantize_rco.py first"
for f in "$name-RCO-IQ3_S.gguf" "$name-RCO-IQ3_XXS.gguf" README.md SHA256SUMS; do
  [ -f "$out/$f" ] || fail "$out/$f not found: run quantize_rco.py with --types IQ3_S,IQ3_XXS"
done

echo "=== checking local files against SHA256SUMS"
(cd "$out" && sha256sum --check --quiet SHA256SUMS) || fail "local files don't match SHA256SUMS"

echo "=== uploading $out to $repo"
hf repos create "$repo" --private --exist-ok
hf upload "$repo" "$out" .

echo "=== checking $repo against SHA256SUMS"
python3 -I - "$repo" "$out/SHA256SUMS" <<'EOF'
import json
import os
import sys
import urllib.request

repo, sums = sys.argv[1:]
base = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
req = urllib.request.Request(f"{base}/api/models/{repo}/tree/main?recursive=true",
                             headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"})
with urllib.request.urlopen(req) as r:
    remote = {e["path"]: e.get("lfs", {}).get("oid") for e in json.load(r)}

bad = []
checked = 0
for line in open(sums):
    sha, path = line.rstrip("\n").split("  ", 1)
    if path not in remote:
        bad.append(f"{path}: missing from the repo")
    elif path.endswith(".gguf"):  # the small text files have no sha256 in the listing, only presence
        checked += 1
        if remote[path] != sha:
            bad.append(f"{path}: repo has {remote[path]}, expected {sha}")
if bad:
    sys.exit("error: upload incomplete:\n  " + "\n  ".join(bad))
print(f"  all files present, {checked} GGUF hashes match")
EOF

echo "Done: https://huggingface.co/$repo"
