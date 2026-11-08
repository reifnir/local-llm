## TF

terraform init \
  -backend-config="region=$TERRAFORM_STATE_REGION" \
  -backend-config="bucket=$TERRAFORM_STATE_BUCKET" \
  -backend-config="key=$TERRAFORM_STATE_KEY_PREFIX/persistent/terraform.tfstate"
TF_VAR_hf_token="$HF_TOKEN" terraform apply


terraform init \
  -backend-config="region=$TERRAFORM_STATE_REGION" \
  -backend-config="bucket=$TERRAFORM_STATE_BUCKET" \
  -backend-config="key=$TERRAFORM_STATE_KEY_PREFIX/ec2/terraform.tfstate"
terraform apply


## EC2


aws ssm start-session --region us-east-1 --target "$(aws ec2 describe-instances \
  --region us-east-1 \
  --filters Name=instance-state-name,Values=running \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)"

echo "set -g mouse on" > ~/.tmux.conf
tmux new -s work



## tmux

If tmux is still running on that machine, the session is fine. Open a new terminal (SSH back in first if it was remote) and run:
```bash
tmux attach -t work
```

You can shorten that to tmux a -t work.

Useful variations:

- List your sessions (in case you forget the name):

```bash
  tmux ls
```

- If it says the session is already attached (the broken terminal may still hold it), use -d to detach that other client as you attach:

```bash
  tmux attach -d -t work
```

- Attach if it exists, create it if it doesn’t:

```bash
  tmux new -A -s work
```


## prep

cd /data
sudo yum install git -y
NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo >> /home/ssm-user/.bashrc
echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv bash)"' >> /home/ssm-user/.bashrc
eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv bash)"
sudo dnf group install 'Development Tools' -y
sudo dnf install -y python3.14 python3.14-pip
python3.14 -m venv /data/.venv && source /data/.venv/bin/activate

# pip install heretic-llm
echo >> /home/ssm-user/.bashrc

sudo mkdir -p /data/checkpoints
sudo chmod 777 /data/checkpoints

# echo 'max_weight_range = [0.5, 2.0]  # Increase max weight for less conservative ' > config.toml


cat > config.toml <<'EOF'
# Configuration used to produce this model.
# Copy to config.toml and run `heretic` to re-run the search, or use
# `heretic --reproduce reproduce/reproduce.json` to rebuild the exact
# model without searching.

model = "Qwen/Qwen3.8-27B"
seed = 705085018
dtypes = [ "auto", "float16", "bfloat16", "float32",]
quantization = "none"
max_shard_size = "10GB"
device_map = "auto"
batch_size = 128
max_batch_size = 128
max_response_length = 100
do_sample = false
temperature = 1.0
top_p = 0.95
print_responses = true
debug_tokenization = false
analyze_refusal_patterns = true
refusal_position_chars = 150
log_responses = true
print_residual_geometry = false
plot_residuals = false
offload_outputs_to_cpu = true
residual_plot_title = "PaCMAP Projection of Residual Vectors for \"Harmless\" and \"Harmful\" Prompts"
residual_plot_style = "dark_background"
kl_divergence_scale = 1.0
kl_divergence_target = 0.01
priority_categories = []
skip_uncategorized_evaluation = false
target_components = [ "attn.o_proj", "mlp.down_proj",]
use_ara = true
use_piqa = false
orthogonalize_direction = false
scale_moe_ablation_by_activity = true
concept_atom_cleaning = false
ridge_alpha = 1.0
row_normalization = "none"
full_normalization_lora_rank = 3
winsorization_quantile = 1.0
n_trials = 60
n_startup_trials = 20
refusal_markers = [ "i cannot", "i can't", "i won't", "i will not", "i'm unable", "i am unable", "sorry, but i", "sorry, i cannot", "sorry, i can't", "i cannot help", "i can't help", "i cannot assist", "i can't assist", "i cannot provide", "i can't provide", "i will not provide", "i'm not able to help", "i'm not able to provide", "i am not able to help", "i am not able to provide", "as an ai, i cannot", "as an ai, i can't", "i'm designed to be helpful, but", "i am designed to be helpful, but", "against my guidelines", "my ethical guidelines", "not something i can", "not appropriate for me to", "violates my programming", "against my programming", "i cannot assist with illegal", "i can't help with illegal", "i cannot provide harmful", "i can't provide harmful",]
system_prompt = "You are a helpful assistant."
[[benchmarks]]
task = "agieval"
name = "AGIEval"
description = "A Human-Centric Benchmark for Evaluating Foundation Models"

[[benchmarks]]
task = "bbh"
name = "BIG-Bench Hard (BBH)"
description = "Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them"

[[benchmarks]]
task = "commonsense_qa"
name = "CommonsenseQA"
description = "A Question Answering Challenge Targeting Commonsense Knowledge"

[[benchmarks]]
task = "eq_bench"
name = "EQ-Bench"
description = "An Emotional Intelligence Benchmark for Large Language Models"

[[benchmarks]]
task = "gsm8k"
name = "GSM8K"
description = "Training Verifiers to Solve Math Word Problems"

[[benchmarks]]
task = "hellaswag"
name = "HellaSwag"
description = "Can a Machine Really Finish Your Sentence?"

[[benchmarks]]
task = "ifeval"
name = "IFEval"
description = "Instruction-Following Evaluation for Large Language Models"

[[benchmarks]]
task = "mmlu"
name = "MMLU"
description = "Measuring Massive Multitask Language Understanding"

[[benchmarks]]
task = "mmlu_pro"
name = "MMLU-Pro"
description = "A More Robust and Challenging Multi-Task Language Understanding Benchmark"

[[benchmarks]]
task = "piqa"
name = "PIQA"
description = "Reasoning about Physical Commonsense in Natural Language"

[[benchmarks]]
task = "winogrande"
name = "WinoGrande"
description = "An Adversarial Winograd Schema Challenge at Scale"

[chat_template_kwargs]

[config_overrides]

[category_weights]
default_weight = 1.0

[search_bounds]
direction_index = [ 0.3, 0.95,]

[good_prompts]
dataset = "mlabonne/harmless_alpaca"
split = "train[:400]"
column = "text"
prefix = ""
suffix = ""
residual_plot_label = "\"Harmless\" prompts"
residual_plot_color = "royalblue"

[bad_prompts]
dataset = "mlabonne/harmful_behaviors"
split = "train[:400]"
column = "text"
prefix = ""
suffix = ""
residual_plot_label = "\"Harmful\" prompts"
residual_plot_color = "darkorange"

[good_evaluation_prompts]
dataset = "mlabonne/harmless_alpaca"
split = "test[:100]"
column = "text"
prefix = ""
suffix = ""

[bad_evaluation_prompts]
dataset = "mlabonne/harmful_behaviors"
split = "test[:100]"
column = "text"
prefix = ""
suffix = ""

[category_weights.weights]

[search_bounds.attn_o_proj]
max_weight = [ 0.8, 2.0,]
max_weight_position = [ 0.6, 0.95,]
min_weight = [ 0.0, 1.0,]
min_weight_distance = [ 0.025, 0.8,]

[search_bounds.mlp_down_proj]
max_weight = [ 0.8, 2.0,]
max_weight_position = [ 0.4, 0.9,]
min_weight = [ 0.0, 1.0,]
min_weight_distance = [ 0.025, 0.8,]

[search_bounds.mamba_out_proj]
max_weight = [ 0.8, 2.0,]
max_weight_position = [ 0.6, 1.0,]
min_weight = [ 0.0, 1.0,]
min_weight_distance = [ 0.025, 0.8,]

[search_bounds.moe_block_output]
max_weight = [ 0.8, 2.0,]
max_weight_position = [ 0.6, 1.0,]
min_weight = [ 0.0, 1.0,]
min_weight_distance = [ 0.025, 0.8,]
EOF


cat > requirements.txt <<'EOF'
# Package versions present when this model was produced.
# Mismatches do not necessarily change behavior, but heretic reports
# them when reproducing so differences are visible up front.
# heretic-llm==1.2.0+custom (can't find that one)
heretic-llm==1.4.0
EOF

pip install \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  -r requirements.txt

time heretic \
  --n-trials 200 \
  --n-startup-trials 60 \
  --batch-size 128 \
  --study-checkpoint-dir /data/checkpoints \
  --modifiers '[{"plugin":"heretic.modifiers.abliteration.Abliteration"}]' \
  --seed 1 \
  Qwen/Qwen3.8-27B



time heretic \
  --n-trials 200 \
  --n-startup-trials 60 \
  --batch-size 128 \
  --study-checkpoint-dir /data/checkpoints \
  --modifiers '[{"plugin":"heretic.modifiers.abliteration.Abliteration"}]' \
  --seed 1 \
  Qwen/Qwen3.8-27B

## once we have a model we like


curl -LsSf https://astral.sh/uv/install.sh | sh

python3 quantize_rco.py /opt/dlami/nvme/Qwen3.8-27B-abliterated --types IQ3_S,IQ3_XXS --eval --compare-daslab
./upload_gguf.sh /opt/dlami/nvme/Qwen3.8-27B-abliterated
