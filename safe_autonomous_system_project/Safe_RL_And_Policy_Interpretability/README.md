# VIPER for SafePO

An implementation of the **VIPER** (Verifiable Imitation Policy via Extracted Rules) algorithm applied to safety-constrained reinforcement learning environments using [SafetyGymnasium](https://github.com/PKU-Alignment/safety-gymnasium) and [SafePO](https://github.com/PKU-Alignment/Safe-Policy-Optimization).

VIPER distills a trained neural network policy (the "oracle") into an interpretable decision tree that approximates its behavior. The resulting tree policy is human-readable, verifiable, and deployable without a neural network at inference time.

---

## How It Works

1. **Oracle** — A neural network policy (`ActorVCritic`) pre-trained with SafePO on a `SafetyCarGoal` task.
2. **DAgger-style data collection** — Trajectories are sampled by mixing the current tree policy and the oracle. On the first iteration, only the oracle is used.
3. **Weighted imitation** — Each state is weighted by an importance score: the spread between the oracle's highest and lowest log-probabilities over a discretized action set. States where the best and worst actions differ most are weighted more heavily.
4. **Tree fitting** — A `DecisionTreeRegressor` is fit to the oracle's actions using the importance weights.
5. **Best policy selection** — After all iterations, the tree with the highest mean episode reward is saved.

---

## Project Structure

```
viper_safepo/
├── viper.py              # Main VIPER training loop
├── evaluate.py           # Policy evaluation loop (safety-gym compatible)
├── monitor.py            # Gym Monitor wrapper for safety-gym envs
├── gym_env/
│   └── __init__.py       # Environment factory (make_env)
├── model/
│   └── tree_wrapper.py   # Sklearn tree wrapped in SB3-compatible interface
└── train/
    └── oracle.py         # Oracle training utilities (DQN/PPO via SB3)
```

---

## Dependencies

- [safety-gymnasium](https://github.com/PKU-Alignment/safety-gymnasium)
- [safepo](https://github.com/PKU-Alignment/Safe-Policy-Optimization)
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- [scikit-learn](https://scikit-learn.org/)
- [PyTorch](https://pytorch.org/)
- numpy, tqdm, joblib, pandas

---

## Usage

### Prerequisites

Ensure a SafePO-trained oracle model exists at:
```
../safepo/runs/<agent>_exp/SafetyCarGoal<level>-v0/<agent>/*/torch_save/model499.pt
../safepo/runs/<agent>_exp/SafetyCarGoal<level>-v0/<agent>/*/config.json
```

### Train a VIPER tree

```bash
python viper.py --agent <agent_name> --level <0|1|2> --verbose <0|1|2>
```

| Argument | Description |
|---|---|
| `-a`, `--agent` | Name of the SafePO agent (e.g. `ppo_lag`) |
| `-l`, `--level` | Goal difficulty level (0, 1, or 2) |
| `-v`, `--verbose` | Verbosity: 0 = reward only, 1 = silent, 2 = reward each iter |

The best tree is saved to:
```
../viper_agents/<agent>.joblib
```

### Key hyperparameters (set in `viper.py`)

| Variable | Default | Description |
|---|---|---|
| `n_iter` | 100 | Number of VIPER iterations |
| `total_timesteps` | 10000 | Total environment steps across all iterations |
| `max_depth` | None | Max depth of the decision tree (None = unlimited) |
| `max_leaves` | None | Max leaf nodes (None = unlimited) |

---

## Output

After training, the script prints:
- Best iteration index
- Best mean reward
- Tree depth and number of leaves
- Save path of the `.joblib` model file
