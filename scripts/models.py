#!/usr/bin/env python3
"""Manage the curated models in models.toml: list, install, verify, accept, select, status, chat.

Runs inside the `tool` container (see compose.yaml) with the repo mounted at /work; the Makefile
drives Docker itself. Standard library only. Every downloaded file is hashed while it streams and
must match the sha256 pinned in models.toml before it is kept.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "models.toml"
MODELS_DIR = ROOT / "models"
STATE_DIR = ROOT / ".state"
VERIFIED_CACHE = STATE_DIR / "verified.json"
ACTIVE_FILE = STATE_DIR / "active.json"
LLAMA_ARGS_FILE = STATE_DIR / "llama.args"
SERVER = "http://llm:8080"  # the llm service on the compose network

HF = "https://huggingface.co"
CHUNK = 8 << 20
SMALL_FILE_LIMIT = 64 << 20

# Formats that can execute code when loaded (pickle) or are code outright. Never downloaded.
UNSAFE_SUFFIXES = (".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".npy", ".npz", ".py", ".pyc", ".so")

SAFETY_LABELS = {"A": "vendor", "B": "3rd-party", "C": "abliterated"}


class Fail(Exception):
    """A user-facing error: printed without a traceback."""


# ── config ──────────────────────────────────────────────────────────────────────


def load_config() -> tuple[dict, dict[str, dict]]:
    with CONFIG.open("rb") as f:
        cfg = tomllib.load(f)
    models: dict[str, dict] = {}
    for m in cfg.get("model", []):
        mid = m.get("id")
        if not mid:
            raise Fail("models.toml: a [[model]] entry has no id")
        if mid in models:
            raise Fail(f"models.toml: duplicate id {mid!r}")
        for key in ("kind", "repo", "revision", "safety"):
            if key not in m:
                raise Fail(f"models.toml: {mid} is missing {key!r}")
        m.setdefault("files", {})
        models[mid] = m
    return cfg.get("settings", {}), models


def get_model(models: dict[str, dict], mid: str | None) -> dict:
    if not mid:
        raise Fail("no model given: use MODEL=<id> (see `make list`)")
    if mid not in models:
        close = [k for k in models if mid.lower() in k.lower()]
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise Fail(f"unknown model {mid!r}.{hint}")
    return models[mid]


def model_dir(m: dict) -> Path:
    return MODELS_DIR / m["repo"]


def is_installed(m: dict) -> bool:
    return bool(m["files"]) and all((model_dir(m) / f).exists() for f in m["files"])


def is_unsafe(path: str) -> bool:
    return path.lower().endswith(UNSAFE_SUFFIXES)


def read_active() -> dict | None:
    try:
        return json.loads(ACTIVE_FILE.read_text())
    except (OSError, ValueError):
        return None


# ── hashing / verification cache ────────────────────────────────────────────────


def _read_cache() -> dict:
    try:
        return json.loads(VERIFIED_CACHE.read_text())
    except (OSError, ValueError):
        return {}


def _write_cache(cache: dict) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    tmp = VERIFIED_CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=1, sort_keys=True))
    os.replace(tmp, VERIFIED_CACHE)


def _stat_key(p: Path) -> list:
    st = p.stat()
    return [st.st_size, st.st_mtime_ns, st.st_ino]


def remember_verified(p: Path, sha: str) -> None:
    cache = _read_cache()
    cache[str(p.relative_to(ROOT))] = {"sha256": sha, "stat": _stat_key(p)}
    _write_cache(cache)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    total = p.stat().st_size
    done, last = 0, 0.0
    with p.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
            done += len(chunk)
            if time.monotonic() - last > 0.5:
                last = time.monotonic()
                print(f"\r  hashing {p.name}  {done / 1e9:6.2f}/{total / 1e9:.2f} GB", end="", flush=True)
    print("\r" + " " * 100 + "\r", end="")
    return h.hexdigest()


def verify_file(p: Path, expected: str, full: bool) -> str:
    """Return 'ok' | 'missing' | 'mismatch'. Trusts the stat cache unless full=True."""
    if not p.exists():
        return "missing"
    if not full:
        hit = _read_cache().get(str(p.relative_to(ROOT)))
        if hit and hit["sha256"] == expected and hit["stat"] == _stat_key(p):
            return "ok"
    if sha256_file(p) != expected:
        return "mismatch"
    remember_verified(p, expected)
    return "ok"


def verify_model(m: dict, full: bool = False, quiet: bool = False) -> bool:
    if not m["files"]:
        raise Fail(f"{m['id']} has no pinned files yet: run `make accept MODEL={m['id']}`")
    ok = True
    for name, sha in m["files"].items():
        status = verify_file(model_dir(m) / name, sha, full)
        if status != "ok" or not quiet:
            print(f"  {status:8} {name}")
        if status == "mismatch":
            print(f"           expected sha256 {sha}\n"
                  f"           the file on disk was changed or replaced: delete it and re-run `make install`")
        ok &= status == "ok"
    return ok


# ── Hugging Face ────────────────────────────────────────────────────────────────


def hf_open(url: str, extra_headers: dict | None = None, timeout: int = 60):
    req = urllib.request.Request(url, headers={"User-Agent": "local-llm/1.0", **(extra_headers or {})})
    token = os.environ.get("HF_TOKEN")
    if token:
        # unredirected: the token goes to huggingface.co only, never to the CDN it redirects to
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise Fail(f"HTTP {e.code} for {url}\n  The repo may be gated: request access on its Hugging Face page "
                       f"and put HF_TOKEN=<token> in .env") from None
        if e.code == 404:
            raise Fail(f"HTTP 404 for {url}\n  The pinned file or revision no longer exists upstream.") from None
        raise


def hf_json(path: str) -> object:
    with hf_open(f"{HF}/api/{path}") as r:
        return json.load(r)


def hf_file_url(repo: str, revision: str, path: str) -> str:
    return f"{HF}/{repo}/resolve/{revision}/{urllib.parse.quote(path)}"


def download(m: dict, name: str, expected: str) -> None:
    """Stream one file to <name>.part (resuming if present), hashing as it goes; keep it only on a match."""
    dest = model_dir(m) / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    have = 0
    if part.exists():
        have = part.stat().st_size
        with part.open("rb") as f:
            while chunk := f.read(CHUNK):
                h.update(chunk)
    try:
        resp = hf_open(hf_file_url(m["repo"], m["revision"], name), {"Range": f"bytes={have}-"} if have else {})
    except urllib.error.HTTPError as e:
        if e.code != 416:  # 416: the .part is already complete
            raise
        resp = None
    if resp is not None:
        with resp:
            if have and resp.status != 206:  # server ignored the range: start over
                have, h = 0, hashlib.sha256()
            total = have + int(resp.headers.get("Content-Length", 0))
            done, t0, last = have, time.monotonic(), 0.0
            with part.open("ab" if have else "wb") as out:
                while chunk := resp.read(CHUNK):
                    out.write(chunk)
                    h.update(chunk)
                    done += len(chunk)
                    if time.monotonic() - last > 0.5:
                        last = time.monotonic()
                        rate = (done - have) / max(last - t0, 1e-6) / 1e6
                        print(f"\r  {name}  {done / 1e9:6.2f}/{total / 1e9:.2f} GB  {rate:6.1f} MB/s",
                              end="", flush=True)
            print("\r" + " " * 100 + "\r", end="")
    actual = h.hexdigest()
    if actual != expected:
        part.unlink()
        raise Fail(f"sha256 mismatch for {m['repo']}/{name}\n  expected {expected}\n  got      {actual}\n"
                   f"  The download was discarded. If upstream changed on purpose, review it with "
                   f"`make accept MODEL={m['id']}`.")
    os.replace(part, dest)
    remember_verified(dest, actual)
    print(f"  verified {name}")


def upstream_pins(m: dict, revision: str) -> dict[str, str]:
    """sha256 of every file the model needs at `revision`."""
    tree = hf_json(f"models/{m['repo']}/tree/{revision}?recursive=true")
    entries = {e["path"]: e for e in tree if e.get("type") == "file"}
    if m.get("snapshot"):
        patterns = m.get("include", [])
        wanted = sorted(p for p in entries if any(fnmatch.fnmatch(Path(p).name, pat) for pat in patterns))
        for p in (p for p in wanted if is_unsafe(p)):
            print(f"  refusing unsafe format: {p}")
        wanted = [p for p in wanted if not is_unsafe(p)]
    else:
        wanted = list(m["files"]) or [f for f in (m.get("gguf"), m.get("mmproj")) if f]
        missing = [p for p in wanted if p not in entries]
        if missing:
            raise Fail(f"not present upstream at {revision[:12]}: {', '.join(missing)}")
    pins = {}
    for path in wanted:
        if is_unsafe(path):
            raise Fail(f"refusing to pin {path}: pickle/code formats are not allowed")
        e = entries[path]
        if e.get("lfs"):
            pins[path] = e["lfs"]["oid"]
        else:  # small git-tracked file: HF only publishes its sha1, so hash the content ourselves
            if e.get("size", 0) > SMALL_FILE_LIMIT:
                raise Fail(f"{path} is large but not stored in LFS; refusing to hash it in memory")
            with hf_open(hf_file_url(m["repo"], revision, path)) as r:
                pins[path] = hashlib.sha256(r.read()).hexdigest()
    return pins


def rewrite_pins(text: str, mid: str, revision: str, pins: dict[str, str]) -> str:
    """Replace `revision` and the `[model.files]` table of one [[model]] block, leaving the rest as is."""
    lines = text.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.strip() == "[[model]]"] + [len(lines)]
    id_re = re.compile(rf'^\s*id\s*=\s*"{re.escape(mid)}"\s*(#.*)?$')
    for a, b in zip(starts, starts[1:]):
        if any(id_re.match(line) for line in lines[a:b]):
            break
    else:
        raise Fail(f"could not find [[model]] block for {mid}")
    rev_i = next((i for i in range(a, b) if re.match(r"^\s*revision\s*=", lines[i])), None)
    files_i = next((i for i in range(a, b) if lines[i].strip() == "[model.files]"), None)
    if rev_i is None or files_i is None:
        raise Fail(f"{mid}: block needs a `revision = ...` line and a `[model.files]` table")
    lines[rev_i] = f'revision = "{revision}"\n'
    end = files_i + 1
    while end < b and lines[end].strip() and not lines[end].lstrip().startswith(("[", "#")):
        end += 1
    lines[files_i + 1:end] = [f'"{name}" = "{sha}"\n' for name, sha in pins.items()]
    return "".join(lines)


# ── llama-server arguments ──────────────────────────────────────────────────────


def llama_args(m: dict, settings: dict, vision: bool, ctx_override: int | None) -> list[str]:
    container_dir = Path("/models") / m["repo"]
    kv = settings.get("kv_cache_type", "q8_0")
    ctx = ctx_override or m.get("ctx", 32768)
    args = [
        "--host", "0.0.0.0", "--port", "8080",
        "--cors-origins", "localhost",  # default allows any web page in your browser to call the server
        "--model", str(container_dir / m["gguf"]),
        "--alias", m["id"],
        "--jinja",
        "--flash-attn", "on",
        "--cache-type-k", kv, "--cache-type-v", kv,
        "--parallel", "1",
        "--threads", str(settings.get("threads", 8)),
        "--fit", "on",
        "--fit-target", str(settings.get("fit_margin_mib", 1024)),
    ]
    if m["placement"] == "gpu":
        # every layer pinned to the GPU; --fit may only size the context, never below ctx
        args += ["--n-gpu-layers", "999", "--fit-ctx", str(ctx)]
    elif m["placement"] == "split":
        # context fixed; --fit decides which layers / MoE expert tensors live in system RAM
        args += ["--ctx-size", str(ctx)]
    else:
        raise Fail(f"{m['id']}: unknown placement {m['placement']!r}")
    if vision and m.get("mmproj"):
        args += ["--mmproj", str(container_dir / m["mmproj"])]
    if m.get("mtp"):
        args += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]
    flags = {"temp": "--temp", "top_p": "--top-p", "top_k": "--top-k", "min_p": "--min-p",
             "presence_penalty": "--presence-penalty"}
    for key, value in m.get("sampling", {}).items():
        if key not in flags:
            raise Fail(f"{m['id']}: unsupported sampling key {key!r}")
        args += [flags[key], str(value)]
    return args


# ── commands ────────────────────────────────────────────────────────────────────


def cmd_list(args) -> None:
    settings, models = load_config()
    active = (read_active() or {}).get("id")
    kinds = {"llm": "LLMs (make use MODEL=<id>)", "tts": "Text-to-speech (make path MODEL=<id>)",
             "asr": "Speech recognition (make path MODEL=<id>)"}
    for kind, title in kinds.items():
        rows = [m for m in models.values() if m["kind"] == kind]
        if not rows:
            continue
        print(f"\n{title}")
        print(f"  {'':2}{'id':30} {'where':6} {'GB':>5}  {'safety':14} {'inst':4}  use")
        for m in rows:
            mark = "*" if m["id"] == active else ("d" if m["id"] == settings.get("default_llm") else "")
            safety = f"{m['safety']} {SAFETY_LABELS.get(m['safety'], '')}"
            print(f"  {mark:2}{m['id']:30} {m.get('placement', '-'):6} {m.get('size_gb', 0):5.1f}  "
                  f"{safety:14} {'yes' if is_installed(m) else '-':4}  {m.get('use', '')}")
    print("\n  * = selected for serving   d = default for `make up`   safety: see the header of models.toml")


def cmd_install(args) -> None:
    _, models = load_config()
    m = get_model(models, args.model)
    if not m["files"]:
        raise Fail(f"{m['id']} has no pinned files yet. Review and pin upstream first: make accept MODEL={m['id']}")
    print(f"{m['id']}: {m['repo']} @ {m['revision'][:12]}  (safety {m['safety']}, {m.get('license', '?')})")
    for name, sha in m["files"].items():
        if is_unsafe(name):
            raise Fail(f"refusing {name}: pickle/code formats are not allowed")
        p = model_dir(m) / name
        if p.exists():
            if verify_file(p, sha, full=False) != "ok":
                raise Fail(f"{p.relative_to(ROOT)} exists but does not match its pinned sha256. "
                           f"Delete it to re-download.")
            print(f"  present  {name}")
            continue
        download(m, name, sha)
    print(f"installed {m['id']}")


def cmd_verify(args) -> None:
    _, models = load_config()
    targets = [get_model(models, args.model)] if args.model else [
        m for m in models.values() if any((model_dir(m) / f).exists() for f in m["files"])]
    if not targets:
        print("nothing installed")
        return
    bad = []
    for m in targets:
        print(f"{m['id']}:")
        if not verify_model(m, full=True):
            bad.append(m["id"])
    if bad:
        raise Fail(f"verification FAILED: {', '.join(bad)}")
    print("all verified against models.toml")


def cmd_accept(args) -> None:
    _, models = load_config()
    m = get_model(models, args.model)
    revision = args.revision or hf_json(f"models/{m['repo']}/revision/main")["sha"]
    print(f"{m['id']}: {m['repo']}\n  pinned   {m['revision']}\n  upstream {revision}")
    new = upstream_pins(m, revision)
    old = m["files"]
    changes = []
    for name in sorted(set(old) | set(new)):
        if name not in new:
            changes.append(f"  - removed  {name}")
        elif name not in old:
            changes.append(f"  + added    {name}  {new[name]}")
        elif old[name] != new[name]:
            changes.append(f"  ~ changed  {name}\n      {old[name]}\n   -> {new[name]}")
    if not changes and revision == m["revision"]:
        print("  pins already match upstream; nothing to do")
        return
    print("\n".join(changes) if changes else "  file hashes unchanged (revision only)")
    if not args.yes:
        if not sys.stdin.isatty():
            raise Fail("not a terminal: re-run with YES=1 to accept without a prompt")
        if input("Accept these pins into models.toml? [y/N] ").strip().lower() != "y":
            raise Fail("not accepted; models.toml unchanged")
    text = rewrite_pins(CONFIG.read_text(), m["id"], revision, new)
    parsed = {x["id"]: x for x in tomllib.loads(text)["model"]}[m["id"]]
    if parsed["revision"] != revision or parsed.get("files", {}) != new:
        raise Fail("internal error: rewritten models.toml did not round-trip; left unchanged")
    CONFIG.write_text(text)
    print(f"pinned {m['id']} @ {revision[:12]} ({len(new)} files). Run `make install MODEL={m['id']}`.")


def cmd_remove(args) -> None:
    _, models = load_config()
    m = get_model(models, args.model)
    if args.serving and (read_active() or {}).get("id") == m["id"]:
        raise Fail(f"{m['id']} is being served; run `make down` first")
    shared = {model_dir(o) / f for o in models.values() if o["id"] != m["id"] for f in o["files"]}
    removed = 0
    for name in m["files"]:
        p = model_dir(m) / name
        for q in (p, p.with_name(p.name + ".part")):
            if q.exists() and q not in shared:
                q.unlink()
                removed += 1
    print(f"removed {removed} file(s) for {m['id']} (files shared with other models were kept)")


def cmd_select(args) -> None:
    """Verify an LLM and write the llama-server arguments; the Makefile then (re)starts the server."""
    settings, models = load_config()
    mid = args.model or (read_active() or {}).get("id") or settings.get("default_llm")
    m = get_model(models, mid)
    if m["kind"] != "llm":
        raise Fail(f"{m['id']} is a {m['kind']} model; only LLMs are served here. "
                   f"Its files are at: make path MODEL={m['id']}")
    if not is_installed(m):
        raise Fail(f"{m['id']} is not installed: run `make install MODEL={m['id']}`")
    print(f"verifying {m['id']} ...")
    if not verify_model(m, full=args.full_verify, quiet=True):
        raise Fail("refusing to serve a model that does not match its pins")
    STATE_DIR.mkdir(exist_ok=True)
    LLAMA_ARGS_FILE.write_text("\n".join(llama_args(m, settings, args.vision, args.ctx)) + "\n")
    ACTIVE_FILE.write_text(json.dumps({"id": m["id"], "vision": args.vision, "ctx": args.ctx}))
    print(f"selected {m['id']} ({m['placement']}, ~{m.get('size_gb', '?')} GB)")


def cmd_path(args) -> None:
    _, models = load_config()
    m = get_model(models, args.model)
    if not is_installed(m):
        raise Fail(f"{m['id']} is not installed: run `make install MODEL={m['id']}`")
    host = Path(os.environ.get("HOST_ROOT", str(ROOT))) / model_dir(m).relative_to(ROOT)
    if m.get("snapshot"):
        print(host)
    else:
        for name in m["files"]:
            print(host / name)


def server_get(path: str, timeout: float = 3.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(f"{SERVER}{path}", timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, ""
    except (urllib.error.URLError, OSError):
        return 0, ""


def cmd_status(args) -> None:
    active = read_active()
    if not active:
        print("llm: none selected (make use MODEL=<id>)")
        return
    code, body = server_get("/props")
    if code == 200:
        props = json.loads(body)
        ctx = props.get("default_generation_settings", {}).get("n_ctx", "?")
        print(f"llm: {active['id']} ready, context {ctx} tokens")
    else:
        print(f"llm: {active['id']} selected, server {'loading' if code == 503 else 'not running'}")


def cmd_claude_env(args) -> None:
    """Print `export` lines that point Claude Code at the served model (stdout only holds the exports)."""
    _, models = load_config()
    active = (read_active() or {}).get("id")
    if not active:
        raise Fail("no model is being served: run `make use MODEL=<id>` first")
    m = get_model(models, active)
    support = m.get("claude_code", "no")
    if support == "no":
        raise Fail(f"{active} cannot be used with Claude Code (claude_code = \"no\" in models.toml)")
    code, body = server_get("/props")
    if code != 200:
        raise Fail(f"{active} is selected but the server is not ready; run `make up` / `make status`")
    n_ctx = int(json.loads(body).get("default_generation_settings", {}).get("n_ctx", 0))
    if support == "limited" or n_ctx < 65536:
        print(f"warning: {active} has a {n_ctx}-token context; Claude Code's base prompt alone is 15-30K tokens, "
              f"so expect frequent compaction", file=sys.stderr)
    if m["safety"] == "C":
        print(f"warning: {active} is abliterated (safety C) and Claude Code gives it a shell: "
              f"only use it on code you trust", file=sys.stderr)
    env = {
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{os.environ.get('LLM_PORT', '8080')}",
        "ANTHROPIC_AUTH_TOKEN": "local",  # any value; makes this the credential instead of your claude.ai login
        # one model serves every role: main loop, aliases, background tasks, subagents
        **{k: active for k in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
                               "ANTHROPIC_DEFAULT_HAIKU_MODEL", "ANTHROPIC_DEFAULT_FABLE_MODEL",
                               "CLAUDE_CODE_SUBAGENT_MODEL")},
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(n_ctx),  # real window, so auto-compaction triggers in time
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(min(32000, max(8192, n_ctx // 4))),
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",  # keeps the system prompt prefix stable for llama.cpp's KV cache
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "API_FORCE_IDLE_TIMEOUT": "0",  # long prompt processing on split models can pause the stream
    }
    print("unset ANTHROPIC_API_KEY")
    for k, v in env.items():
        print(f"export {k}={v}")


def cmd_chat(args) -> None:
    active = (read_active() or {}).get("id")
    body = json.dumps({"model": active, "max_tokens": args.max_tokens,
                       "messages": [{"role": "user", "content": args.prompt}]}).encode()
    req = urllib.request.Request(f"{SERVER}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            res = json.load(r)
    except (urllib.error.URLError, OSError) as e:
        raise Fail(f"server not reachable ({e}); is it running? `make status`") from None
    msg = res["choices"][0]["message"]
    if msg.get("reasoning_content"):
        print(f"[thinking]\n{msg['reasoning_content'].strip()}\n[/thinking]\n")
    print(msg.get("content", "").strip())
    t = res.get("timings", {})
    if t:
        print(f"\n-- {active}: prompt {t.get('prompt_per_second', 0):.0f} tok/s, "
              f"generation {t.get('predicted_per_second', 0):.1f} tok/s ({t.get('predicted_n', 0)} tokens)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    for name, fn in (("install", cmd_install), ("path", cmd_path)):
        sp = sub.add_parser(name)
        sp.add_argument("model")
        sp.set_defaults(fn=fn)
    sp = sub.add_parser("remove")
    sp.add_argument("model")
    sp.add_argument("--serving", action="store_true", help="the llm container is running")
    sp.set_defaults(fn=cmd_remove)
    sp = sub.add_parser("verify")
    sp.add_argument("model", nargs="?")
    sp.set_defaults(fn=cmd_verify)
    sp = sub.add_parser("accept")
    sp.add_argument("model")
    sp.add_argument("--revision", help="commit to pin (default: current main)")
    sp.add_argument("--yes", action="store_true")
    sp.set_defaults(fn=cmd_accept)
    sp = sub.add_parser("select")
    sp.add_argument("model", nargs="?")
    sp.add_argument("--vision", action="store_true", help="load the vision projector (~1 GB VRAM)")
    sp.add_argument("--ctx", type=int, help="override the context size from models.toml")
    sp.add_argument("--full-verify", action="store_true", help="re-hash files even if unchanged on disk")
    sp.set_defaults(fn=cmd_select)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    sub.add_parser("claude-env").set_defaults(fn=cmd_claude_env)
    sp = sub.add_parser("chat")
    sp.add_argument("prompt")
    sp.add_argument("--max-tokens", type=int, default=2048)
    sp.set_defaults(fn=cmd_chat)
    args = p.parse_args()
    try:
        args.fn(args)
    except Fail as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
