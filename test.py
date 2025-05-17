import os
import random
from typing import Callable, List, Type
import gymnasium as gym
import numpy as np
import torch


def evaluate(
    model_path: str,
    make_env_fn: Callable[[str, int, int, bool, str], Callable],
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: Type[torch.nn.Module],
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True
) -> List[float]:
    """
    Evaluate a DQN model on a given Gym environment.

    Args:
        model_path (str): Path to the trained model weights (.pth).
        make_env_fn (Callable): Function to create the evaluation environment.
        env_id (str): Gym environment ID.
        eval_episodes (int): Number of episodes to evaluate.
        run_name (str): Directory name for video saving (if enabled).
        Model (torch.nn.Module): Model class (not an instance).
        device (torch.device): Torch device ("cpu" or "cuda").
        epsilon (float): Probability of taking a random action (exploration).
        capture_video (bool): Whether to record evaluation video.

    Returns:
        List[float]: List of episodic returns over evaluation episodes.
    """
    envs = gym.vector.SyncVectorEnv([make_env_fn(env_id, 0, 0, capture_video, run_name)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []

    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:   #epsilon greedy exploration.
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                q_values = model(torch.tensor(obs, dtype=torch.float32).to(device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, _, _, _, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" in info:
                    print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                    episodic_returns.append(info["episode"]["r"])

        obs = next_obs

    envs.close()
    return episodic_returns


if __name__ == "__main__":

    from main import DeepQModel, build_env  # Assumes correct import path and names

    # Configuration
    env_id = "ALE/MsPacman-v5"
    model_path = "./runs/ALE/MsPacman-v5__MsPacman-v5__1__1746914087/MsPacman-v5.pth"
    device = torch.device("cpu")  # Or torch.device("cuda") if GPU is available
    episodes = 100
    epsilon = 0.05
    capture_video = False

    # Evaluation
    returns = evaluate(
        model_path=model_path,
        make_env_fn=build_env,
        env_id=env_id,
        eval_episodes=episodes,
        run_name="eval",
        Model=DeepQModel,
        device=device,
        epsilon=epsilon,
        capture_video=capture_video
    )

    avg_return = np.mean(returns)
    print(f"Average return over {episodes} episodes: {avg_return:.2f}")

