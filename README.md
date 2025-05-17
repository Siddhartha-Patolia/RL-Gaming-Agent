# RL Gaming Agent using Deep Q-Network (DQN) for Atari Games 
This repository provides a PyTorch implementation of the Deep Q-Network (DQN) algorithm for training agents in classic Atari environments using the Gymnasium API. It supports features like experience replay, target networks, epsilon-greedy exploration, and evaluation with video recording. Logging is supported via TensorBoard and Weights & Biases (WandB).

---

##  Features

- DQN architecture with convolutional neural networks
- Experience Replay Buffer
- Target network updates (soft update)
- Linear epsilon-greedy exploration
- Evaluation with optional video capture
- Experiment tracking via TensorBoard and WandB
- Uses Stable Baselines3 Atari wrappers for preprocessing

---

## Files

- `main.py`: Main training script for DQN on Atari environments.
- `test.py`: Evaluation script to test trained models and optionally record performance videos.
- `results/`: Folder where videos from evaluation are saved (if enabled).

---

## To run use these Command Line Arguments:
python3 main.py \
    --exp-name MsPacman-v5 \
    --track \
    --gamma 0.95 \
    --end-e 0.05 \
    --exploration-fraction 0.15 \
    --cuda True\
    --wandb-project-name ALE \
    --capture-video \
    --env-id ALE/MsPacman-v5 \
    --total-timesteps 5000000 \
    --buffer-size 400000 \
    --save-model
