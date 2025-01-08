# Cartpole PPO

## Requirements

Install `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

```bash
uv venv # Create a virtual environment
uv pip install -r pyproject.toml # Install dependencies
```

## Train

```bash
uv run train.py
```

## Evaluate

```bash
uv run evaluate.py
```

The idea is to train the agent with PPO and then evaluate it.

## Tensorboard

```bash
uv run -m tensorboard.main --logidr ./tensorboard_logs
```