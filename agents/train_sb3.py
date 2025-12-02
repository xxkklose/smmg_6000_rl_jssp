# agents/train_sb3.py
import argparse
import os
import glob
import random
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import torch
import json
import shutil
from tqdm import tqdm
from envs.jobshop_env import JobShopEnv, generate_random_instance, Job

def make_env_from_file(fname, reward_shaping="sparse"):
    from utils.dataset import load_instance, load_jsp_instance
    if fname.endswith('.jsp'):
        jobs, n_machines, best_make_span, best_schedule = load_jsp_instance(fname)
    else:
        jobs, n_machines = load_instance(fname)
    return JobShopEnv(jobs=jobs, n_machines=n_machines, reward_shaping=reward_shaping, max_steps=1000)

def make_env_from_generated(n_jobs=5, n_machines=3, seed=0, reward_shaping="sparse"):
    jobs = generate_random_instance(n_jobs=n_jobs, n_machines=n_machines, max_ops=4, seed=seed)
    return JobShopEnv(jobs=jobs, n_machines=n_machines, reward_shaping=reward_shaping, max_steps=1000)

def train(args):
    os.makedirs(args.logdir, exist_ok=True)

    # Resolve device string to a torch.device, with sensible fallbacks on macOS (MPS) or CPU
    def _resolve_device(dev_str: str):
        # prefer explicit torch.device when possible
        if dev_str is None:
            return torch.device("cpu")
        dev = dev_str.lower()
        if dev == "cuda":
            if torch.cuda.is_available():
                print("Using CUDA device")
                return torch.device("cuda")
            # try MPS (Apple Silicon) if available
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                print("CUDA not available; using MPS device")
                return torch.device("mps")
            print("CUDA not available; falling back to CPU")
            return torch.device("cpu")
        if dev == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                print("Using MPS device")
                return torch.device("mps")
            print("MPS not available; falling back to CPU")
            return torch.device("cpu")
        # any other string (e.g., 'cpu') -> let torch interpret it
        try:
            return torch.device(dev)
        except Exception:
            return torch.device("cpu")

    resolved_device = _resolve_device(args.device)
    args.device = resolved_device

    # Support training from glob of JSP/JSON instances
    class DatasetWrapper(gym.Env):
        def __init__(self, files, reward_shaping="sparse"):
            self.files = files
            # bootstrap spaces using first file
            bootstrap_env = make_env_from_file(self.files[0], reward_shaping)
            self.observation_space = bootstrap_env.observation_space
            self.action_space = bootstrap_env.action_space
            self.reward_shaping = reward_shaping
            self._env = bootstrap_env

        def reset(self, *, seed=None, options=None):
            f = random.choice(self.files)
            self._env = make_env_from_file(f, reward_shaping=self.reward_shaping)
            obs, info = self._env.reset(seed=seed)
            return obs, info

        def step(self, action):
            return self._env.step(action)

        def render(self, mode="human"):
            return self._env.render(mode)

    if args.train_glob:
        train_files = sorted(glob.glob(args.train_glob))
        if len(train_files) == 0:
            raise FileNotFoundError(f"No files match train_glob: {args.train_glob}")
        env = DatasetWrapper(train_files, reward_shaping=args.reward_shaping)
    elif args.use_dataset:
        env = make_env_from_file(args.instance_file, reward_shaping=args.reward_shaping)
    else:
        env = make_env_from_generated(n_jobs=args.n_jobs, n_machines=args.n_machines, seed=0, reward_shaping=args.reward_shaping)

    def _make_vec_env(base_env_or_files):
        n_envs = max(1, int(args.n_envs))
        if isinstance(base_env_or_files, list):
            fns = [lambda: DatasetWrapper(base_env_or_files, reward_shaping=args.reward_shaping) for _ in range(n_envs)]
        else:
            fns = [lambda: base_env_or_files for _ in range(n_envs)]
        vec = SubprocVecEnv(fns) if n_envs > 1 else DummyVecEnv(fns)
        vec = VecNormalize(vec, norm_obs=True, norm_reward=False)
        return vec

    vec_env = None
    if args.train_glob and len(args.train_glob) > 0:
        train_files = sorted(glob.glob(args.train_glob))
        vec_env = _make_vec_env(train_files)
    else:
        vec_env = _make_vec_env(env)

    algo = args.algo.lower()
    model = None
    policy_kwargs = dict(net_arch=[args.net_size, args.net_size])

    if algo == "dqn":
        model = DQN("MlpPolicy", vec_env,
                    learning_rate=1e-3,
                    buffer_size=50000,
                    learning_starts=100,
                    batch_size=256,
                    gamma=0.99,
                    train_freq=4,
                    tau=1.0,
                    target_update_interval=1000,
                    verbose=1,
                    policy_kwargs=policy_kwargs,
                    device=args.device)
    elif algo == "ppo":
        model = PPO("MlpPolicy", vec_env,
                    learning_rate=3e-4,
                    n_steps=256,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    device=args.device)
    elif algo == "sac":
        # SAC is off-policy continuous action by default; action space is discrete here.
        # stable-baselines3's SAC expects Box actions. However we can still try to train a continuous policy that outputs
        # a probability distribution over discrete actions using a custom wrapper — to keep it simple we use PPO/DQN for discrete.
        # But SB3 contains a Categorical DQN; for completeness we still provide SAC branch for continuous relaxations.
        model = SAC("MlpPolicy", vec_env,
                    learning_rate=3e-4,
                    buffer_size=100000,
                    batch_size=64,
                    tau=0.005,
                    gamma=0.99,
                    verbose=1,
                    policy_kwargs=policy_kwargs,
                    device=args.device)
    else:
        raise ValueError("Unsupported algo")

    # Callbacks
    if os.path.exists(args.logdir):
        shutil.rmtree(args.logdir)
    os.makedirs(args.logdir, exist_ok=True)
    checkpoint_cb = CheckpointCallback(save_freq=100000, save_path=args.logdir, name_prefix=f"{algo}_model")
    callbacks = [checkpoint_cb]
    if args.val_glob:
        val_files = sorted(glob.glob(args.val_glob))
        if len(val_files) > 0:
            eval_env = DummyVecEnv([lambda: DatasetWrapper(val_files, reward_shaping=args.reward_shaping)])
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
            eval_cb = EvalCallback(eval_env, best_model_save_path=args.logdir, log_path=args.logdir,
                                   eval_freq=2000, deterministic=True, render=False)
            callbacks.append(eval_cb)
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks if len(callbacks)>1 else checkpoint_cb, progress_bar=True)
    model.save(os.path.join(args.logdir, f"{algo}_final"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn","ppo","sac"])
    parser.add_argument("--logdir", type=str, default="./logs")
    parser.add_argument("--total_timesteps", type=int, default=20000)
    parser.add_argument("--use_dataset", action="store_true")
    parser.add_argument("--instance_file", type=str, default="datasets/instance_0.json")
    parser.add_argument("--train_glob", type=str, default="")
    parser.add_argument("--val_glob", type=str, default="")
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--net_size", type=int, default=128)
    parser.add_argument("--n_jobs", type=int, default=5)
    parser.add_argument("--n_machines", type=int, default=3)
    parser.add_argument("--reward_shaping", type=str, default="dense")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    train(args)
