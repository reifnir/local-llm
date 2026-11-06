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
pip install heretic-llm
echo >> /home/ssm-user/.bashrc

sudo mkdir -p /data/checkpoints
sudo chmod 777 /data/checkpoints
echo 'max_weight_range = [0.5, 2.0]  # Increase max weight for less conservative ' > config.toml
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
