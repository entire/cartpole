# Cartpole PPO

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