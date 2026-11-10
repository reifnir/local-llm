# Local models for this machine: hash-pinned downloads (models.toml) and a llama.cpp server.
# Everything runs in Docker; `make help` lists the targets, `make help <target>` describes one.

SHELL := /bin/bash
.DEFAULT_GOAL := help
SELF := $(lastword $(MAKEFILE_LIST))

-include .env
LLM_PORT ?= 8080
export LLM_PORT HF_TOKEN
export HOST_UID := $(shell id -u)
export HOST_GID := $(shell id -g)
export HOST_ROOT := $(CURDIR)

COMPOSE := docker compose
TOOL := $(COMPOSE) --progress quiet --profile tool run --rm tool
TOOL_PIPE := $(COMPOSE) --progress quiet --profile tool run --rm -T tool # no TTY: output is captured
BENCH := $(COMPOSE) --progress quiet --profile tool run --rm -T --entrypoint python tool scripts/bench.py
SANDBOX := $(COMPOSE) --progress quiet --profile tool run --rm -T sandbox

# Positional model ids: `make install gemma4-12b` == `make install MODEL=gemma4-12b`
POSITIONAL := install accept verify remove path use bench
ifneq ($(filter $(firstword $(MAKECMDGOALS)),$(POSITIONAL)),)
  ifneq ($(word 2,$(MAKECMDGOALS)),)
    MODEL := $(word 2,$(MAKECMDGOALS))
    $(eval $(MODEL):;@:)
  endif
endif

# `make help status use`: the words after `help` are topics, not goals to run.
ifeq ($(firstword $(MAKECMDGOALS)),help)
  TOPICS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
endif

# Each public target has a `## summary` on its line and `## ` detail lines just above it;
# `##@ Group` lines head the groups in `make help`. All read by scripts/make-help.awk.

ifneq ($(TOPICS),)
.PHONY: help $(TOPICS)
$(filter-out help,$(TOPICS)):
	@:
else

.PHONY: list install path remove use up down status logs chat claude claude-env bench bench-report verify accept pull init-persistent init-ec2 _tf_check

##@ Models

## Usage:   make list
## Inputs:  none (reads models.toml and .state/)
## Output:  one table each for LLMs, text-to-speech, speech recognition and source weights.
##          Columns: id, placement (gpu / split), download size in GB, safety grade (A / B / C),
##          published by (the Hugging Face account the files come from), installed, and what it's
##          for. `*` marks the selected LLM, `d` the default for `make up`.
list: ## Table of curated models (installed, selected, safety, publisher)
	@$(TOOL) list

## Usage:   make install <id>        (or MODEL=<id>)
## Inputs:  MODEL (required): an id from `make list`. HF_TOKEN in .env, only for gated repos.
## Output:  the repo, pinned revision, safety grade and license; then live progress (GB, MB/s)
##          per file, and "verified <file>" once its sha256 matches ("present" if already
##          downloaded and verified). Ends with "installed <id>". Files go to models/<repo>/.
##          A sha256 mismatch deletes the download and fails. Interrupted downloads resume.
install: ## Download MODEL, verifying every file against its pinned sha256
	@$(TOOL) install $(MODEL)

## Usage:   make path <id>
## Inputs:  MODEL (required; must be installed)
## Output:  host paths, one per line: the model directory for TTS/ASR models (pass it to
##          from_pretrained), or each GGUF file for LLMs.
path: ## Print where MODEL's files are (for using TTS/ASR models from other projects)
	@$(TOOL) path $(MODEL)

## Usage:   make remove <id>
## Inputs:  MODEL (required)
## Output:  "removed N file(s) for <id>". Files another model also pins are kept, and the
##          entry stays in models.toml. Refuses while <id> is being served (`make down` first).
remove: ## Delete MODEL's downloaded files
	@$(TOOL) remove $(MODEL) $(if $(shell $(COMPOSE) ps -q llm 2>/dev/null),--serving)

##@ Serving

## Usage:   make use <id> [VISION=1] [CTX=<tokens>] [FULL=1]
## Inputs:  MODEL (required; an installed LLM).
##          VISION=1: also load the image projector (~1 GB VRAM; models with an mmproj only).
##          CTX: context size instead of models.toml's `ctx` (the minimum, for gpu models).
##          FULL=1: re-hash every file instead of trusting the stat cache.
## Output:  "verifying <id>", "selected <id> (<placement>, ~<GB> GB)", a wait while the model
##          loads, then `make status` and the OpenAI / Anthropic / web UI URLs. Replaces whatever
##          was being served. If the server fails to start: its last 40 log lines, and exit 1.
use: ## Serve MODEL with llama.cpp (stops whatever was being served)
	@$(TOOL) select $(MODEL) $(if $(VISION),--vision) $(if $(CTX),--ctx $(CTX)) $(if $(FULL),--full-verify)
	@$(MAKE) --no-print-directory _start

## Usage:   make up
## Inputs:  none. Serves the model from the last `make use` (same VISION / CTX), or
##          settings.default_llm from models.toml if none was ever selected.
## Output:  the same as `make use`.
up: ## Start serving the last-used model (or settings.default_llm)
	@$(TOOL) select
	@$(MAKE) --no-print-directory _start

_start:
	@echo "loading (large models can take a minute) ..."
	@$(COMPOSE) up -d --force-recreate --wait llm \
	  || { $(COMPOSE) logs --tail 40 llm; echo "llama-server failed to start (logs above). A 'gpu' model that does not fit fails here: free VRAM or pick a 'split' model."; exit 1; }
	@$(MAKE) --no-print-directory status
	@echo ""
	@echo "  OpenAI API:    http://127.0.0.1:$(LLM_PORT)/v1   (any API key)"
	@echo "  Anthropic API: http://127.0.0.1:$(LLM_PORT)/v1/messages"
	@echo "  Web UI:        http://127.0.0.1:$(LLM_PORT)"

## Usage:   make down
## Inputs:  none
## Output:  Docker's progress as the container is removed. Frees the VRAM. Models and the
##          selection are kept, so `make up` brings the same model back.
down: ## Stop and remove the server container (models stay on disk)
	@$(COMPOSE) --profile tool down --remove-orphans

## Usage:   make status
## Inputs:  none
## Output:  up to three lines:
##            llm: running (Up 5 minutes (healthy))      container state; absent if stopped
##            llm: <id> ready, context <n> tokens        or: selected, server loading / not running
##            gpu: <used> MiB / <total> MiB used         whole GPU, including Windows' share
status: ## What is being served, and GPU memory
	@$(COMPOSE) ps --format '{{.Service}}: {{.State}} ({{.Status}})' llm 2>/dev/null
	@$(TOOL) status
	@command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader \
	  | awk -F', ' '{ print "gpu: " $$1 " / " $$2 " used" }' || true

## Usage:   make logs
## Inputs:  none
## Output:  llama-server's log, followed live until Ctrl-C: model loading, where tensors were
##          placed, and timings for each request.
logs: ## Follow llama-server logs
	@$(COMPOSE) logs -f llm

## Usage:   make chat [PROMPT="..."]
## Inputs:  PROMPT (optional; default: asks for a Python palindrome function). Needs the server.
## Output:  the model's reasoning between [thinking] and [/thinking] (if it thinks), its answer,
##          then "-- <id>: prompt N tok/s, generation N tok/s (N tokens)".
chat: ## Quick test prompt with tokens/s (PROMPT="...")
	@$(TOOL) chat "$${PROMPT:-Write a Python function that checks whether a string is a palindrome.}"

##@ Claude Code

## Usage:   make claude [ARGS="<claude arguments>"]
## Inputs:  ARGS (optional): passed through to `claude`. Needs the server and the claude CLI.
## Output:  an interactive Claude Code session in this repo, backed by the served model.
##          For other projects, symlink scripts/claude-local onto your PATH and run it there.
claude: ## Launch Claude Code here against the served model; elsewhere use scripts/claude-local
	@scripts/claude-local $(ARGS)

## Usage:   eval "$(make -s claude-env)"
## Inputs:  none. Needs the server, serving a model with claude_code = "yes" or "limited".
## Output:  on stdout, `unset ANTHROPIC_API_KEY` and `export` lines: ANTHROPIC_BASE_URL,
##          ANTHROPIC_AUTH_TOKEN, the model variables, CLAUDE_CODE_MAX_CONTEXT_TOKENS (the real
##          context) and a few more. Warnings (small context, abliterated model) go to stderr.
##          Exits 1 if nothing usable is being served.
claude-env: ## Print env vars pointing Claude Code at the served model: eval "$(make -s claude-env)"
	@$(TOOL_PIPE) claude-env

##@ Benchmarks

## Usage:   make bench [<id>] [ALL=1] [LIMIT=<n>|all]
## Inputs:  MODEL (optional; default: the selected model). ALL=1: every installed LLM in turn.
##          LIMIT: HumanEval problems per model (default 40, from bench/bench.toml; all = 164).
## Output:  per model: load time, prompt (4K, 24K) and generation tok/s, a line per problem
##          (tokens, seconds), then "HumanEval passed/n = pass@1". Appends to
##          bench/results.jsonl and regenerates bench/results.md; completions and failures stay
##          in bench/runs/<run>/. Restores what was being served. At 40 problems: ~25 min for a
##          gpu model, 1.5-2 h for a split one.
bench: ## Benchmark speed + HumanEval for MODEL, the served model, or ALL=1 installed chat models
	@set -e; \
	prev=$$($(BENCH) active 2>/dev/null || true); \
	was_up=$$($(COMPOSE) ps -q llm 2>/dev/null); \
	if [ -n "$(ALL)" ]; then models=$$($(BENCH) installed); else models="$(or $(MODEL),$$prev)"; fi; \
	[ -n "$$models" ] || { echo "nothing to benchmark: pass MODEL=<id> or ALL=1"; exit 1; }; \
	for m in $$models; do $(MAKE) --no-print-directory _bench_one BM=$$m; done; \
	if [ -n "$$was_up" ] && [ "$$prev" != "$$m" ]; then \
	  $(MAKE) --no-print-directory use MODEL=$$prev >/dev/null 2>&1 && echo "restored: serving $$prev" || echo "could not restore $$prev: run make use MODEL=$$prev"; \
	elif [ -z "$$was_up" ]; then \
	  [ -z "$$prev" ] || $(TOOL_PIPE) select $$prev >/dev/null; $(COMPOSE) --progress quiet rm -sf llm >/dev/null; \
	fi
	@echo "results: bench/results.md"

# One model: fresh start (timed), VRAM delta, speed + HumanEval completions, sandboxed tests, record.
_bench_one:
	@set -e; echo "=== $(BM)"; \
	$(TOOL_PIPE) select $(BM) >/dev/null; \
	$(COMPOSE) --progress quiet rm -sf llm >/dev/null; \
	base=$$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || true); \
	t0=$${EPOCHREALTIME/./}; \
	$(COMPOSE) --progress quiet up -d --wait llm || { $(COMPOSE) logs --tail 30 llm; exit 1; }; \
	load_ms=$$(( ($${EPOCHREALTIME/./} - t0) / 1000 )); \
	after=$$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || true); \
	vram=""; if [ -n "$$base" ] && [ -n "$$after" ]; then vram="--vram-mib $$((after - base))"; fi; \
	run=$$($(BENCH) run --model $(BM) --load-ms $$load_ms $$vram $(if $(LIMIT),--limit $(LIMIT))); \
	echo "running $$(wc -l < bench/runs/$$run/completions.jsonl) solutions in the sandbox ..."; \
	$(SANDBOX) /runs/$$run/completions.jsonl > bench/runs/$$run/eval.jsonl; \
	$(BENCH) record $$run

## Usage:   make bench-report
## Inputs:  none (reads bench/results.jsonl)
## Output:  rewrites bench/results.md: the newest run per model and problem count, best pass@1
##          first. Only needed after editing results.jsonl by hand; `make bench` does this.
bench-report: ## Regenerate bench/results.md from bench/results.jsonl
	@$(BENCH) report

##@ Integrity

## Usage:   make verify [<id>]
## Inputs:  MODEL (optional; default: every model with files on disk)
## Output:  for each model, "ok", "missing" or "mismatch" per file, with progress while hashing.
##          Always re-hashes every byte (ignores the stat cache `make use` trusts). Ends with
##          "all verified against models.toml", or "verification FAILED: <ids>" and exit 1.
verify: ## Re-hash installed files (all, or just MODEL) against models.toml
	@$(TOOL) verify $(MODEL)

## Usage:   make accept <id> [REV=<commit>] [YES=1]
## Inputs:  MODEL (required). REV: the Hugging Face commit to pin (default: upstream main).
##          YES=1 accepts without the [y/N] prompt (required when not run from a terminal).
## Output:  the pinned and upstream revisions, then each added (+), removed (-) or changed (~)
##          file with its sha256, or "pins already match upstream". If you accept, the
##          model's revision and [model.files] are rewritten in models.toml: review and commit
##          that diff. Downloads nothing; run `make install` afterwards.
accept: ## Review upstream changes for MODEL and re-pin revision + sha256 in models.toml
	@$(TOOL) accept $(MODEL) $(if $(REV),--revision $(REV)) $(if $(YES),--yes)

##@ Terraform

TF_STATE_VARS := TERRAFORM_STATE_REGION TERRAFORM_STATE_BUCKET TERRAFORM_STATE_KEY_PREFIX
TF_BACKEND = -backend-config="region=$(TERRAFORM_STATE_REGION)" \
  -backend-config="bucket=$(TERRAFORM_STATE_BUCKET)" \
  -backend-config="key=$(TERRAFORM_STATE_KEY_PREFIX)/$(1)/terraform.tfstate"

_tf_check:
	@$(foreach v,$(TF_STATE_VARS),[ -n "$($(v))" ] || { echo "$(v) is not set (export it or add it to .env)"; exit 1; };)

## Usage:   make init-persistent
## Inputs:  TERRAFORM_STATE_REGION, TERRAFORM_STATE_BUCKET, TERRAFORM_STATE_KEY_PREFIX, from
##          the environment or .env.
## Output:  `terraform init` for terraform/persistent against the S3 state key
##          <prefix>/persistent/terraform.tfstate. Applies nothing; afterwards run
##          `TF_VAR_hf_token="$HF_TOKEN" terraform apply` in terraform/persistent.
init-persistent: ## terraform init for terraform/persistent (S3 backend)
	@$(MAKE) --no-print-directory _tf_check
	terraform -chdir=terraform/persistent init $(call TF_BACKEND,persistent)

## Usage:   make init-ec2
## Inputs:  TERRAFORM_STATE_REGION, TERRAFORM_STATE_BUCKET, TERRAFORM_STATE_KEY_PREFIX, from
##          the environment or .env.
## Output:  `terraform init` for terraform/ec2 against the S3 state key
##          <prefix>/ec2/terraform.tfstate. Applies nothing; afterwards run `terraform apply`
##          in terraform/ec2.
init-ec2: ## terraform init for terraform/ec2 (S3 backend)
	@$(MAKE) --no-print-directory _tf_check
	terraform -chdir=terraform/ec2 init $(call TF_BACKEND,ec2)

# RunInstances can't queue for spot capacity, so retry the whole apply while AWS reports none.
SPOT_CAPACITY_ERRORS := InsufficientInstanceCapacity|InsufficientCapacity
RETRY_DELAY ?= 60

## Usage:   make apply-ec2 [RETRY_DELAY=<seconds>]
## Inputs:  AWS credentials; terraform/ec2 already initialized (`make init-ec2`).
##          RETRY_DELAY: seconds between attempts (default 60).
## Output:  `terraform apply -auto-approve` in terraform/ec2 (no approval prompt), re-run
##          after RETRY_DELAY whenever it fails with a spot capacity error, until it
##          succeeds. Stops on any other error. Ctrl-C to give up.
apply-ec2: ## terraform apply for terraform/ec2, retrying until spot capacity is available
	@log=$$(mktemp); trap 'rm -f "$$log"' EXIT; attempt=1; \
	until terraform -chdir=terraform/ec2 apply -auto-approve -input=false 2>&1 | tee "$$log"; [ "$${PIPESTATUS[0]}" -eq 0 ]; do \
	  grep -Eq '$(SPOT_CAPACITY_ERRORS)' "$$log" || exit 1; \
	  echo "apply-ec2: no spot capacity (attempt $$attempt), retrying in $(RETRY_DELAY)s" >&2; \
	  attempt=$$((attempt + 1)); sleep $(RETRY_DELAY); \
	done

##@ Other

## Usage:   make pull
## Inputs:  none
## Output:  Docker's pull progress for the digest-pinned llama.cpp and python images. Optional:
##          other targets pull them on first use; this just does it up front.
pull: ## Pre-pull the pinned Docker images
	@$(COMPOSE) --profile tool pull

endif

## Usage:   make help [<target>...]
## Inputs:  target names (optional)
## Output:  without arguments, every target with a one-line summary and the common options;
##          with targets, each one's usage, inputs and output.
.PHONY: help
help: ## List targets; `make help <target>` describes one
ifeq ($(TOPICS),)
	@echo "usage: make <target> [<id> | MODEL=<id>] [OPTION=value ...]"
	@awk -f scripts/make-help.awk $(SELF)
	@echo ""
	@echo "options: VISION=1  CTX=<tokens>  FULL=1  (use)    REV=<commit>  YES=1  (accept)"
	@echo "         ALL=1  LIMIT=<n>|all  (bench)    PROMPT=\"...\"  (chat)    ARGS=\"...\"  (claude)"
	@echo "details: make help <target>    e.g. make help use"
else
	@first=1; for t in $(TOPICS); do \
	  [ -n "$$first" ] || echo; first=; \
	  awk -v topic=$$t -f scripts/make-help.awk $(SELF) || exit 1; \
	done
endif
