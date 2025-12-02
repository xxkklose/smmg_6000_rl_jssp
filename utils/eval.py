# utils/eval.py
import numpy as np
import os
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from stable_baselines3 import DQN, PPO, SAC
from envs.jobshop_env import JobShopEnv
from utils.dataset import load_instance, load_jsp_instance

def plot_gantt_chart(subject, n_machines, save_path="gantt_chart.png"):
    fig, ax = plt.subplots(figsize=(10, 6))

    if hasattr(subject, "jobs"):
        jobs_list = subject.jobs
    else:
        jobs_list = subject

    num_jobs = len(jobs_list) if jobs_list is not None else 0
    colors = plt.get_cmap('tab20', max(1, num_jobs))

    if hasattr(subject, "timeline") and getattr(subject, "timeline"):
        timeline = subject.timeline
        for item in timeline:
            if isinstance(item, (list, tuple)):
                if len(item) >= 4:
                    job_id, machine_id, start_time, finish_time = item[0], item[1], item[2], item[3]
                    op_index = item[4] if len(item) >= 5 else 0
                    duration = float(finish_time) - float(start_time)
                    ax.broken_barh([(float(start_time), duration)], (int(machine_id)*10, 9),
                                   facecolors=colors(int(job_id) % max(1, num_jobs)), edgecolors='black')
                    ax.text(float(start_time) + duration/2, int(machine_id)*10 + 4.5,
                            f'J{int(job_id)}-O{int(op_index)}', ha='center', va='center', color='white', fontsize=8)
                else:
                    continue
            else:
                continue
    else:
        for j_idx, job in enumerate(jobs_list):
            start_time = 0.0
            for op_idx, (machine_id, processing_time) in enumerate(job.ops):
                end_time = start_time + processing_time
                ax.broken_barh([(start_time, processing_time)], (machine_id*10, 9),
                               facecolors=(colors(j_idx % max(1, num_jobs))), edgecolors='black')
                ax.text(start_time + processing_time/2, machine_id*10 + 4.5,
                        f'J{job.job_id}-O{op_idx}', ha='center', va='center', color='white', fontsize=8)
                start_time = end_time

    ax.set_yticks([i*10 + 4.5 for i in range(n_machines)])
    ax.set_yticklabels([f'Machine {i}' for i in range(n_machines)])
    ax.set_xlabel('Time')
    ax.set_title('Gantt Chart')
    plt.grid(True)
    out_dir = os.path.dirname(save_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model_path, algo, instance_file, n_episodes=10, render=False):
    if instance_file.endswith('.jsp'):
        jobs, n_machines, best_makespan, best_schedule = load_jsp_instance(instance_file)
    else:
        jobs, n_machines = load_instance(instance_file)
    env = JobShopEnv(jobs=jobs, n_machines=n_machines, reward_shaping="sparse", max_steps=1000)
    model = None
    if algo == "dqn":
        model = DQN.load(model_path)
    elif algo == "ppo":
        model = PPO.load(model_path)
    elif algo == "sac":
        model = SAC.load(model_path)
    else:
        raise ValueError()
    
    name_list = instance_file.split("/")
    dataset_name = name_list[-2]
    instance_name = name_list[-1].split(".")[0]
    print(f"Evaluating {algo} on {dataset_name}/{instance_name}")

    makespans = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done=False
        while not done:
            # if model deterministic
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            if render:
                env.render()
        # compute makespan; some jobs might (unexpectedly) have None completion_time
        times = [j.completion_time for j in env.jobs if j.completion_time is not None]
        if times:
            ms = max(times)
        else:
            # fallback: use environment's current simulated time
            # this covers edge-cases where completion_time wasn't set for jobs
            ms = getattr(env, "current_time", 0.0)
        makespans.append(ms)
        print(f"Episode {ep+1}: Makespan = {ms}")
        # Pass env to plotter to use timeline if present
        if not os.path.exists(f"logs/plot/{dataset_name}/{instance_name}"):
            os.makedirs(f"logs/plot/{dataset_name}/{instance_name}")
        plot_gantt_chart(env, n_machines, save_path=f"logs/plot/{dataset_name}/{instance_name}/{algo}_gantt_chart_ep{ep+1}.png")
    return np.mean(makespans), np.std(makespans)

def map_global_to_index(global_index, n_machines):
    job_id = global_index // n_machines
    op_index = global_index % n_machines
    return job_id, op_index

def plot_best_gantt_chart(data_path):
    jobs, n_machines, best_makespan, best_schedule = load_jsp_instance(data_path)
    name_list = data_path.split("/")
    dataset_name = name_list[-2]
    instance_name = name_list[-1].split(".")[0]
    out_dir = f"logs/plot/{dataset_name}/{instance_name}"
    os.makedirs(out_dir, exist_ok=True)
    env = JobShopEnv(jobs=jobs, n_machines=n_machines, reward_shaping="sparse", max_steps=1000)
    # Convert per-machine global indices schedule into timeline entries with dummy timing using ops durations
    timeline = []
    machine_time = [0.0 for _ in range(n_machines)]
    for m_id, seq in enumerate(best_schedule):
        for g_idx in seq:
            j_id, op_idx = map_global_to_index(int(g_idx), n_machines)
            if j_id < len(jobs) and op_idx < len(jobs[j_id].ops):
                op = jobs[j_id].ops[op_idx]
                proc_m, proc_t = int(op[0]), float(op[1])
                start = machine_time[m_id]
                finish = start + proc_t
                timeline.append((int(j_id), int(m_id), float(start), float(finish), int(op_idx)))
                machine_time[m_id] = finish
    env.load_best_schedule(timeline)
    plot_gantt_chart(env, n_machines, save_path=f"{out_dir}/gantt_chart_best.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--algo", type=str, required=True)
    parser.add_argument("--instance", type=str, default="datasets/instance_0.json")
    args = parser.parse_args()
    mean, std = evaluate_model(args.model_path, args.algo, args.instance, n_episodes=5, render=False)
    print(f"Mean makespan: {mean:.3f} ± {std:.3f}")
    # plot_best_gantt_chart("benchmarks/validation/10x10_1.jsp")
