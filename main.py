import gymnasium as gym

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import argparse
import os
import random
import time
from distutils.util import strtobool
import wandb

from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    MaxAndSkipEnv,
    EpisodicLifeEnv,
    NoopResetEnv
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import stable_baselines3 as sb3

from test import evaluate

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)

    parser.add_argument("--env-id", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--erb-size", type=int, default=1000000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=1.)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--target-network-frequency", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eps-max", type=float, default=1.0)
    parser.add_argument("--eps-min", type=float, default=0.01)
    parser.add_argument("--exploration-fraction", type=float, default=0.10)
    parser.add_argument("--intial-steps", type=int, default=80000)
    parser.add_argument("--total-timesteps", type=int, default=10000000)
    args = parser.parse_args()

    # can be increased so that the agent learns from multiple enviroments simultaenously. 
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    return args

def build_env(env_name, env_idx, seed_val , sess_name , rec_vid):
    def create():
        if rec_vid and env_idx == 0:
            environment = gym.make(env_name, render_mode="rgb_array")
            environment = gym.wrappers.RecordVideo(environment, f"videos/{sess_name}")
        else:
            environment = gym.make(env_name)

        environment = gym.wrappers.RecordEpisodeStatistics(environment)
        environment = NoopResetEnv(environment, noop_max=30)
        environment = MaxAndSkipEnv(environment, skip=4)
        environment = EpisodicLifeEnv(environment)
        environment = ClipRewardEnv(environment)
        environment = gym.wrappers.ResizeObservation(environment, (84, 84))
        environment = gym.wrappers.GrayScaleObservation(environment)
        environment = gym.wrappers.FrameStack(environment, 4)
        environment.action_space.seed(seed_val)
        return environment

    return create

class DeepQModel(nn.Module):
    def __init__(self, environment):
        super().__init__()
        action_count = environment.single_action_space.n
        self.model = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_count),
        )

    def forward(self, input_tensor):
        return self.model(input_tensor / 255.0)

def epsilon_decay(e_start: float, e_final: float, total_steps: int, current_step: int):
    decay_rate = (e_final - e_start) / total_steps
    return max(e_start + decay_rate * current_step, e_final)

if __name__ == "__main__":
    args = get_config()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [build_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = DeepQModel(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = DeepQModel(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.erb_size,  
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False
    )
    start_time = time.time()

    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        epsilon = epsilon_decay(args.eps_max, args.eps_min, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, rewards, terminated, truncated, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episode_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)

        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncated):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

        obs = next_obs

        if global_step > args.intial_steps:  # ✅ FIXED TYPED ARG NAME
            if global_step % args.train_freq == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.pth"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

        episodic_returns = evaluate(
            model_path,
            build_env,
            args.env_id,
            eval_episode=10,
            run_name=f"{run_name}-eval",
            Model=DeepQModel,
            device=device,
            epsilon=0.05,
        )

        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

    envs.close()
    writer.close()
