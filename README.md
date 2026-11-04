# local-llm

Curated, hash-pinned local models for one machine (RTX 5080 16 GB, Ryzen 7 9800X3D, 64 GB RAM,
Windows + WSL2), with a Makefile to download, verify, switch, and serve them.

- **LLMs** for software development are served by llama.cpp on `http://127.0.0.1:8080`
  (OpenAI- and Anthropic-compatible).
- **TTS / ASR models** (voice cloning, transcription) are only downloaded and verified here, for
  other projects to load from disk.

Everything runs in Docker. The host needs Docker (with the NVIDIA runtime) and `make`.

## Quick start

```bash
make list                         # the curated table: size, placement, safety, publisher, installed
make install qwen3.6-35b-a3b      # download + sha256-verify (same as MODEL=qwen3.6-35b-a3b)
make use qwen3.6-35b-a3b          # serve it; switching models is just another `make use`
make chat                         # quick prompt, shows tokens/s
make down                         # tear down (models stay on disk); `make up` brings the last one back
```

| Target | What it does |
|---|---|
| `list` | Curated models; `*` = selected, `d` = default for `make up` |
| `install MODEL` | Download, streaming every byte through sha256; mismatches are discarded |
| `path MODEL` | Where the files are on disk (for TTS/ASR use elsewhere) |
| `remove MODEL` | Delete a model's files (shared files are kept) |
| `use MODEL` | Verify, then (re)start llama-server with that model. `VISION=1`, `CTX=n`, `FULL=1` |
| `up` / `down` | Start the last-used model / stop and remove the container |
| `status`, `logs`, `chat` | Health + context size + VRAM / follow logs / test prompt (`PROMPT="..."`) |
| `claude` / `claude-env` | Run Claude Code against the served model / print its env vars (see below) |
| `bench [MODEL]` | Speed + HumanEval benchmark (`ALL=1`, `LIMIT=n\|all`); see below |
| `verify [MODEL]` | Full re-hash of installed files |
| `accept MODEL` | Show upstream changes and re-pin `revision` + sha256 in `models.toml` (`REV=`, `YES=1`) |
| `help [TARGET]` | All targets, or one target's usage, inputs and output (`make help use`) |

## Benchmarks

```bash
make bench qwen3.6-35b-a3b           # one model (default: the one being served)
make bench ALL=1                     # every installed chat model, one after another
make bench gemma4-12b LIMIT=all      # all 164 HumanEval problems instead of 40
```

Each model is started fresh. The benchmark records:
- **Load:** load time and the VRAM it takes.
- **Speed:** prompt-processing speed for a 4K and a 24K-token prompt (about Claude Code's first
  request), and generation speed (median of 3).
- **Quality:** HumanEval pass@1, plus average tokens and seconds per problem. Thinking tokens
  count, which is where ThinkingCap should show its savings.

Results append to [bench/results.jsonl](bench/results.jsonl), and
[bench/results.md](bench/results.md) is regenerated with the newest run per model. Both are worth
committing to track changes across llama.cpp upgrades. Raw completions and per-problem failures
stay in `bench/runs/` (git-ignored). Whatever was being served beforehand is restored at the end.

How long it takes: thinking models spend 1–8K tokens per problem. At the default 40 problems,
expect about 25 min for `gemma4-12b` (~80 tok/s) and 1.5–2 h for split models (~25 tok/s).
`ALL=1` is an overnight job.

Integrity and isolation:
- **Dataset:** pinned by commit and sha256 in [bench/bench.toml](bench/bench.toml).
- **Running generated code:** model-written code only runs in the `sandbox` container. It has no
  network, a read-only filesystem, no capabilities, runs as `nobody` with capped
  memory/CPU/processes, and sees only the completions file. A self-test confirmed that HumanEval's
  reference solutions pass 164/164, and that wrong answers, infinite loops, network access and
  writes all fail.
- **Noise:** 40 problems give roughly ±8 points of noise, so only compare models at the same
  `LIMIT`.

## How integrity works

Each `[[model]]` in [models.toml](models.toml) pins a Hugging Face **commit** (`revision`) and
the **sha256 of every file** (`[model.files]`).

- `install` downloads from that exact commit and keeps a file only if its hash matches.
- `use` re-checks before serving. It trusts a cached result while a file's size, mtime and inode
  are unchanged. `FULL=1` or `make verify` forces a full re-hash.
- `accept` is the only thing that changes a pin. It fetches upstream, prints what was added,
  removed or changed, and rewrites that model's block after you confirm. Commit the diff to
  record the decision. To add a model, write its block with an empty `[model.files]` table and run
  `make accept` on it.
- Pickle formats (`.bin`, `.pt`, `.pth`, `.ckpt`, ...) and repository code (`.py`) are refused
  even if listed. GGUF and safetensors are data-only formats.
- The Docker images are pinned by digest: llama.cpp b12004 (CUDA 12.8), and python:3.13-slim for
  the management tool.

**Safety grades** (also in the `models.toml` header):
- **A**: vendor weights, packaged by the vendor or ggml-org.
- **B**: re-quantized by unsloth, or fine-tuned by ThinkingCap (bottlecapai).
- **C**: abliterated (refusals removed, by huihui-ai). A C model will follow instructions
  injected through files, web pages or issues, so don't give it shell or tool access over
  untrusted content.

## GPU-only vs. split placement

About 13.3 GB of the 16 GB is free while the Windows desktop is running.

- **`gpu`**: every layer stays in VRAM, and llama.cpp grows the context to fill whatever VRAM is
  left, down to the model's `ctx`. If the model can't fit at that context, startup fails instead
  of silently slowing down.
- **`split`**: the context is fixed. llama.cpp's `--fit` keeps attention and dense weights on the
  GPU and moves MoE expert tensors to system RAM until the model fits. MoE models only use about
  3B parameters per token, so they stay fast. The dense `qwen3.8-27b` moves whole layers to RAM
  and decodes noticeably slower.

Measured on this machine (2026-11-01):

| Model | Placement | Generation | Prompt processing |
|---|---|---|---|
| `gemma4-12b` | gpu | 78 tok/s | ~3,800 tok/s |
| `qwen3.6-35b-a3b` | split (experts of 17/41 layers in RAM) | 21-24 tok/s | ~870 tok/s |

MTP speculative decoding made the split MoE model slower (17.9 vs 23.9 tok/s), because drafted
tokens pull in extra experts from RAM. It is off for those models. `--threads 16` was also
slower than 8 (12.9 tok/s).

`make use` ends with `make status`: the context size llama.cpp chose and VRAM in use. `make logs` shows the full load log.

## Using the server

- OpenAI-compatible: `http://127.0.0.1:8080/v1`. Any API key works. The model name is the
  `id` from `make list`.
- Anthropic Messages: `http://127.0.0.1:8080/v1/messages`.
- Web chat UI: `http://127.0.0.1:8080`.

The port is published on localhost only. Windows apps reach it through WSL/Docker Desktop
localhost forwarding. CORS only allows localhost origins, so ordinary web pages in your browser
can't call it.

## Claude Code with a local model

```bash
make use qwen3.6-35b-a3b
ln -s ~/dev/local-llm/scripts/claude-local ~/.local/bin/claude-local   # once
cd ~/some/project && claude-local                                     # any claude args work
```

`claude-local` (or `make claude` inside this repo) reads the served model and its real context
size, sets the environment below, and starts `claude`. Your normal `claude` command is not
affected. To set the variables some other way, e.g. in another shell or tool:
`eval "$(make -s -C ~/dev/local-llm claude-env)"`.

- `ANTHROPIC_BASE_URL` points at llama-server. `ANTHROPIC_AUTH_TOKEN=local` makes it the
  credential, so your claude.ai login and usage aren't used.
- `ANTHROPIC_MODEL`, the opus/sonnet/haiku/fable defaults and `CLAUDE_CODE_SUBAGENT_MODEL` are
  all set to the one served model, so background tasks and subagents hit it too.
- `CLAUDE_CODE_MAX_CONTEXT_TOKENS` is the server's real context, so auto-compaction triggers in
  time.
- `CLAUDE_CODE_ATTRIBUTION_HEADER=0` keeps the system-prompt prefix identical between requests.
  llama.cpp then reuses its KV cache instead of re-processing 15-30K tokens every turn: in
  testing, later turns processed only 23-373 new tokens.
- `API_FORCE_IDLE_TIMEOUT=0` tolerates long prompt processing on split models.
  `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` turns off telemetry and update checks.

The `claude_code` field in `models.toml` marks each model `yes`, `limited` (context too small
for comfortable sessions) or `no` (TTS/ASR). `claude-local` refuses `no` and warns about
`limited` and abliterated models. Claude Code's base prompt is 15-30K tokens, so use 64K+ of
context. Anthropic doesn't support Claude Code on non-Claude models, and agent quality depends
heavily on the model.

## Using the TTS / ASR models from another project

```bash
make install omnivoice && make path omnivoice     # -> /home/jim/dev/local-llm/models/k2-fsa/OmniVoice
```

Load them from that directory, for example `OmniVoice.from_pretrained("<path>")` or
`Qwen3TTSModel.from_pretrained("<path>")`. Set `HF_HUB_OFFLINE=1` so the other project can't
fetch unpinned files. If it runs in Docker, mount `models/` read-only.

Each entry's `library` field says which package loads it:

- `omnivoice` needs `transformers>=5.3`.
- `qwen-tts` pins `transformers==4.57.3`.

They can't share one Python environment.

For mispronunciations:
- OmniVoice accepts inline CMUdict phonemes, e.g. `[B EY1 S] guitar`.
- Qwen3-TTS only supports respelling words.
- `whisper-large-v3-turbo` can transcribe reference clips and round-trip-check output for
  misheard words.

## Machine notes

- **Turn off CUDA sysmem fallback.** In NVIDIA Control Panel → Manage 3D settings, set
  *CUDA – Sysmem Fallback Policy* to *Prefer No Sysmem Fallback*. Otherwise the Windows driver
  can silently page VRAM into system RAM, which makes a `gpu` model crawl instead of failing.
- **WSL memory.** WSL is capped at 56 GB (plus 32 GB swap) by `%UserProfile%\.wslconfig`, leaving
  about 8 GB for Windows. Serving needs little of it (the largest split model uses about 12 GB of
  RAM); the headroom is for loading BF16 safetensors with `device_map="auto"`, e.g. to abliterate
  Qwen3.8-27B (~54 GB: ~12 GB on the GPU, the rest in RAM). Run `wsl --shutdown` after editing it.
- **Why llama.cpp rather than Ollama.** Expert-level MoE offload (`--fit`), loading
  ThinkingCap/huihui GGUFs straight from Hugging Face with sha256 pins, and MTP speculative
  decoding are all first-class in llama.cpp. Ollama runs the same engine underneath.
- **Gated repos.** None of the current entries are gated. If you add one, copy `.env.example` to
  `.env` and set `HF_TOKEN`.
