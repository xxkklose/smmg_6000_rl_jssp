# utils/eval.py
import os
from typing import Tuple, Dict

import gymnasium as gym  # kept for compatibility
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from stable_baselines3 import DQN, PPO, SAC

from envs.jobshop_env import JobShopEnv
from utils.dataset import load_instance, load_jsp_instance


def plot_gantt_chart(subject, n_machines: int, save_path: str = "gantt_chart.png") -> None:
    """
    Plot a Gantt chart.

    `subject` can be:
    - a JobShopEnv-like object with attributes `jobs` and optional `timeline`, or
    - a list of Job objects with `job_id` and `ops`.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if hasattr(subject, "jobs"):
        jobs_list = subject.jobs
    else:
        jobs_list = subject

    num_jobs = len(jobs_list) if jobs_list is not None else 0
    colors = plt.get_cmap("tab20", max(1, num_jobs))

    # Case 1: explicit timeline available (env.run / load_best_schedule)
    if hasattr(subject, "timeline") and getattr(subject, "timeline"):
        timeline = subject.timeline
        for item in timeline:
            # Accept tuple/list or dict
            if isinstance(item, dict):
                job_id = int(item.get("job_id", 0))
                machine_id = int(item.get("machine_id", 0))
                start_time = float(item.get("start", 0.0))
                finish_time = float(item.get("finish", item.get("end", start_time)))
                op_index = int(item.get("op_index", 0))
            else:
                vals = list(item)
                if len(vals) < 4:
                    continue
                job_id = int(vals[0])
                machine_id = int(vals[1])
                start_time = float(vals[2])
                finish_time = float(vals[3])
                op_index = int(vals[4]) if len(vals) >= 5 else 0

            duration = finish_time - start_time
            if duration <= 0:
                continue

            num_jobs_safe = max(1, num_jobs)
            ax.broken_barh(
                [(start_time, duration)],
                (machine_id * 10, 9),
                facecolors=colors(job_id % num_jobs_safe),
                edgecolors="black",
            )
            ax.text(
                start_time + duration / 2.0,
                machine_id * 10 + 4.5,
                f"J{job_id}-O{op_index}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
    else:
        # Case 2: no timeline; naive per-job sequential plot
        for j_idx, job in enumerate(jobs_list):
            start_time = 0.0
            for op_idx, (machine_id, processing_time) in enumerate(job.ops):
                end_time = start_time + processing_time
                num_jobs_safe = max(1, num_jobs)
                ax.broken_barh(
                    [(start_time, processing_time)],
                    (machine_id * 10, 9),
                    facecolors=colors(j_idx % num_jobs_safe),
                    edgecolors="black",
                )
                ax.text(
                    start_time + processing_time / 2.0,
                    machine_id * 10 + 4.5,
                    f"J{job.job_id}-O{op_idx}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )
                start_time = end_time

    ax.set_yticks([i * 10 + 4.5 for i in range(n_machines)])
    ax.set_yticklabels([f"Machine {i}" for i in range(n_machines)])
    ax.set_xlabel("Time")
    ax.set_title("Gantt Chart")
    ax.grid(True)

    out_dir = os.path.dirname(save_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


# def evaluate_model(
#     model_path: str,
#     algo: str,
#     instance_file: str,
#     n_episodes: int = 10,
#     render: bool = False,
# ) -> Tuple[float, float]:
#     """
#     Evaluate a trained RL model (DQN/PPO/SAC) on a given instance and
#     return mean and std of makespan over `n_episodes`.
#     """
#     if instance_file.endswith(".jsp"):
#         jobs, n_machines, best_makespan, best_schedule = load_jsp_instance(instance_file)
#     else:
#         jobs, n_machines = load_instance(instance_file)

#     env = JobShopEnv(
#         jobs=jobs,
#         n_machines=n_machines,
#         reward_shaping="sparse",
#         max_steps=1000,
#     )

#     algo = algo.lower()
#     if algo == "dqn":
#         model = DQN.load(model_path)
#     elif algo == "ppo":
#         model = PPO.load(model_path)
#     elif algo == "sac":
#         model = SAC.load(model_path)
#     else:
#         raise ValueError(f"Unknown algo: {algo}")

#     name_list = instance_file.split("/")
#     dataset_name = name_list[-2] if len(name_list) >= 2 else "dataset"
#     instance_name = name_list[-1].split(".")[0]

#     print(f"Evaluating RL algo '{algo}' on {dataset_name}/{instance_name}")

#     # --- NEW: text log file for RL eval ---
#     log_dir = os.path.join("logs", "text_eval", dataset_name)
#     os.makedirs(log_dir, exist_ok=True)
#     log_path = os.path.join(log_dir, f"{instance_name}_{algo}.txt")
#     log_f = open(log_path, "a", encoding="utf-8")
#     log_f.write(f"Evaluating RL algo '{algo}' on {dataset_name}/{instance_name}\n")
#     log_f.flush()
#     # --------------------------------------

#     makespans = []
#     for ep in range(n_episodes):
#         obs, _ = env.reset()
#         done = False

#         while not done:
#             action, _ = model.predict(obs, deterministic=True)
#             obs, reward, done, truncated, info = env.step(int(action))
#             if render:
#                 env.render()

#         # compute makespan; some jobs might (unexpectedly) have None completion_time
#         times = [j.completion_time for j in env.jobs if j.completion_time is not None]
#         if times:
#             ms = max(times)
#         else:
#             # fallback: use environment's current simulated time
#             ms = getattr(env, "current_time", 0.0)

#         makespans.append(ms)
#         line = f"[RL {algo}] Episode {ep + 1}: Makespan = {ms}\n"
#         print(line.strip())
#         log_f.write(line)
#         log_f.flush()

#         out_dir = f"logs/plot/{dataset_name}/{instance_name}"
#         os.makedirs(out_dir, exist_ok=True)
#         plot_gantt_chart(
#             env,
#             n_machines,
#             save_path=f"{out_dir}/{algo}_gantt_chart_ep{ep + 1}.png",
#         )

#     arr = np.array(makespans, dtype=float)
#     mean_ms = float(arr.mean())
#     std_ms = float(arr.std())
#     summary = f"[RL {algo}] Mean makespan over {n_episodes} episodes: {mean_ms:.3f} ± {std_ms:.3f}\n"
#     print(summary.strip())
#     log_f.write(summary)
#     log_f.close()
#     return mean_ms, std_ms

def evaluate_model(model_path, algo, instance_file, n_episodes=5, render=False):
    """
    Evaluate a trained SB3 model on a given instance (JSON or .jsp),
    log per-episode makespans, and plot Gantt charts.

    This version adds sanity checks so we can see if the RL env is
    consistent with the benchmark (.jsp) data.
    """
    if algo.lower() == "dqn":
        model = DQN.load(model_path)
    elif algo.lower() == "ppo":
        model = PPO.load(model_path)
    elif algo.lower() == "sac":
        model = SAC.load(model_path)
    else:
        raise ValueError("Unsupported algo")

    # ----------------- load instance -----------------
    if instance_file.endswith(".jsp"):
        jobs, n_machines, best_makespan, best_schedule = load_jsp_instance(instance_file)
    else:
        jobs, n_machines = load_instance(instance_file)
        best_makespan, best_schedule = None, None

    name_list = instance_file.split("/")
    dataset_name = name_list[-2] if len(name_list) >= 2 else "dataset"
    instance_name = name_list[-1].split(".")[0]

    print(f"Evaluating RL algo '{algo}' on {dataset_name}/{instance_name}")

    # --- NEW: debug lower bounds from the data we actually loaded ---
    # (this will tell us immediately if the env is seeing "toy" times or the real 10x10 JSP)
    job_sums = [sum(float(p) for (_, p) in j.ops) for j in jobs]
    machine_sums = [0.0 for _ in range(n_machines)]
    for j in jobs:
        for (m, p) in j.ops:
            machine_sums[int(m)] += float(p)
    lb_jobs = max(job_sums) if job_sums else 0.0
    lb_machs = max(machine_sums) if machine_sums else 0.0
    print(
        f"[DEBUG] From loaded instance: LB_jobs={lb_jobs}, "
        f"LB_machines={lb_machs}, best_makespan_from_file={best_makespan}"
    )
    # -------------------------------------------------

    env = JobShopEnv(
        jobs=jobs,
        n_machines=n_machines,
        reward_shaping="sparse",
        max_steps=1000,
    )

    # --- text log file for RL eval ---
    log_dir = os.path.join("logs", "text_eval", dataset_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{instance_name}_{algo}.txt")
    log_f = open(log_path, "w", encoding="utf-8")
    header = (
        f"Evaluating RL algo '{algo}' on {dataset_name}/{instance_name}\n"
        f"LB_jobs={lb_jobs}, LB_machines={lb_machs}, best_makespan_from_file={best_makespan}\n"
    )
    log_f.write(header)
    log_f.flush()
    # -----------------------------

    makespans = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            if render:
                env.render()

        # compute makespan from job completion times
        times = [j.completion_time for j in env.jobs if j.completion_time is not None]
        if times:
            ms = max(times)
        else:
            ms = float(getattr(env, "current_time", 0.0))

        # also compute from timeline to catch any inconsistency
        if getattr(env, "timeline", None):
            timeline_max = max(float(ft) for (_, _, _, ft, _) in env.timeline)
        else:
            timeline_max = float(env.current_time)

        makespans.append(ms)

        debug_line = (
            f"[DEBUG] Ep{ep + 1}: ms_from_jobs={ms}, "
            f"env.current_time={env.current_time}, "
            f"timeline_max={timeline_max}, timeline_len={len(env.timeline)}\n"
        )
        print(debug_line.strip())
        log_f.write(debug_line)

        line = f"[RL {algo}] Episode {ep + 1}: Makespan = {ms}\n"
        print(line.strip())
        log_f.write(line)
        log_f.flush()

        out_dir = f"logs/plot/{dataset_name}/{instance_name}"
        os.makedirs(out_dir, exist_ok=True)
        # put the makespan directly into the filename so you *know* which chart matches which number
        ms_tag = int(round(ms))
        gantt_path = f"{out_dir}/{algo}_gantt_ep{ep + 1}_ms{ms_tag}.png"
        plot_gantt_chart(
            env,
            n_machines,
            save_path=gantt_path,
        )
        print(f"[INFO] Saved Gantt for ep{ep + 1} to {gantt_path}")

    arr = np.array(makespans, dtype=float)
    mean_ms = float(arr.mean())
    std_ms = float(arr.std())
    summary = f"[RL {algo}] Mean makespan over {n_episodes} episodes: {mean_ms:.3f} ± {std_ms:.3f}\n"
    print(summary.strip())
    log_f.write(summary)
    log_f.close()
    return mean_ms, std_ms




def map_global_to_index(global_index: int, n_machines: int) -> Tuple[int, int]:
    """
    Map Taillard/Lawrence-style global operation index to (job_id, op_index)
    assuming operations are laid out job-major with n_machines ops per job.
    """
    job_id = global_index // n_machines
    op_index = global_index % n_machines
    return job_id, op_index


def plot_best_gantt_chart(data_path: str) -> None:
    """
    Plot the reference (best-known) schedule for a .jsp instance using the
    best schedule encoded in the file.
    """
    jobs, n_machines, best_makespan, best_schedule = load_jsp_instance(data_path)
    name_list = data_path.split("/")
    dataset_name = name_list[-2] if len(name_list) >= 2 else "dataset"
    instance_name = name_list[-1].split(".")[0]

    out_dir = f"logs/plot/{dataset_name}/{instance_name}"
    os.makedirs(out_dir, exist_ok=True)

    env = JobShopEnv(
        jobs=jobs,
        n_machines=n_machines,
        reward_shaping="sparse",
        max_steps=1000,
    )

    # Convert per-machine global indices schedule into timeline entries with dummy timing
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


# ---------------------------------------------------------------------------
#  Baseline (non-RL) dispatching rules: SPT, FIFO, EDD, Random
# ---------------------------------------------------------------------------

def _select_heuristic_action(
    env: JobShopEnv,
    rule: str,
    arrival_times: Dict[int, float],
    due_dates: Dict[int, float] or None,
) -> int:
    """
    Choose an action index for JobShopEnv according to a dispatching rule.

    - env._get_valid_action_mask() returns (mask, ready),
      where ready[i] = (job_id, machine_id, proc_time).
    - rule: 'spt' | 'fifo' | 'edd' | 'random'
    - arrival_times[job_id]: when the current operation of that job became available
    - due_dates[job_id]: synthetic or real due date, used only for EDD
    """
    mask, ready = env._get_valid_action_mask()
    # indices where mask == 1.0
    valid_indices = [i for i, v in enumerate(mask) if v > 0.0]
    if not valid_indices:
        # degenerate case: nothing ready; fallback to 0
        return 0

    rule = rule.lower()

    if rule == "spt":
        # Shortest processing time among ready operations
        best_idx = min(valid_indices, key=lambda i: ready[i][2])

    elif rule == "fifo":
        # First-In-First-Out: earliest arrival time of the current operation
        best_idx = min(
            valid_indices,
            key=lambda i: arrival_times.get(ready[i][0], 0.0),
        )

    elif rule == "edd":
        # Earliest Due Date: smallest due date; if not provided, fall back to FIFO
        if due_dates:
            best_idx = min(
                valid_indices,
                key=lambda i: due_dates.get(ready[i][0], float("inf")),
            )
        else:
            best_idx = min(
                valid_indices,
                key=lambda i: arrival_times.get(ready[i][0], 0.0),
            )

    else:
        # Random dispatch for sanity check
        import random
        best_idx = random.choice(valid_indices)

    return int(best_idx)


def evaluate_baseline(
    rule: str,
    instance_file: str,
    n_episodes: int = 10,
    render: bool = False,
) -> Tuple[float, float]:
    """
    Evaluate a non-RL dispatching rule (SPT/FIFO/EDD/Random) on one instance using
    the same JobShopEnv as the RL agents.

    Returns mean and std of makespan over `n_episodes`.
    """
    rule = rule.lower()

    if instance_file.endswith(".jsp"):
        jobs, n_machines, best_makespan, best_schedule = load_jsp_instance(instance_file)
    else:
        jobs, n_machines = load_instance(instance_file)

    # Synthetic due dates for EDD: sum of processing times per job
    due_dates: Dict[int, float] or None = None
    if rule == "edd":
        due_dates = {}
        for j in jobs:
            total_p = float(sum(float(p) for (_, p) in j.ops))
            due_dates[j.job_id] = total_p

    env = JobShopEnv(
        jobs=jobs,
        n_machines=n_machines,
        reward_shaping="sparse",
        max_steps=1000,
    )

    name_list = instance_file.split("/")
    dataset_name = name_list[-2] if len(name_list) >= 2 else "dataset"
    instance_name = name_list[-1].split(".")[0]

    print(f"Evaluating baseline rule '{rule.upper()}' on {dataset_name}/{instance_name}")

        # NEW: text log for baseline eval
    log_dir = os.path.join("logs", "text_eval", dataset_name)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{instance_name}_{rule}.txt")
    log_f = open(log_path, "a", encoding="utf-8")
    log_f.write(f"Evaluating baseline rule '{rule.upper()}' on {dataset_name}/{instance_name}\n")
    log_f.flush()


    makespans = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False

        # For FIFO we need arrival times of current ops; first ops are ready at t=0
        arrival_times: Dict[int, float] = {job.job_id: 0.0 for job in env.jobs}
        # Track previous next_op_idx to detect which job just finished
        prev_next_op = {job.job_id: job.next_op_idx for job in env.jobs}

        while not done:
            action_idx = _select_heuristic_action(env, rule, arrival_times, due_dates)
            obs, reward, done, truncated, info = env.step(int(action_idx))
            if render:
                env.render()

            # Update arrival_times: detect which job's operation just completed
            for job in env.jobs:
                if job.next_op_idx > prev_next_op[job.job_id] and not job.is_done():
                    # Its *next* operation becomes available at current_time
                    arrival_times[job.job_id] = float(env.current_time)
                prev_next_op[job.job_id] = job.next_op_idx

            if truncated:
                break

        times = [j.completion_time for j in env.jobs if j.completion_time is not None]
        if times:
            ms = max(times)
        else:
            ms = getattr(env, "current_time", 0.0)
        makespans.append(ms)

        line = f"[{rule.upper()}] Episode {ep + 1}: Makespan = {ms}\n"
        print(line.strip())
        log_f.write(line)
        log_f.flush()



        print(f"[{rule.upper()}] Episode {ep + 1}: Makespan = {ms}")

        out_dir = f"logs/plot/{dataset_name}/{instance_name}"
        os.makedirs(out_dir, exist_ok=True)
        plot_gantt_chart(
            env,
            n_machines,
            save_path=f"{out_dir}/{rule}_gantt_chart_ep{ep + 1}.png",
        )

    arr = np.array(makespans, dtype=float)
    mean_ms = float(arr.mean())
    std_ms = float(arr.std())

    summary = f"[{rule.upper()}] Mean makespan over {n_episodes} episodes: {mean_ms:.3f} ± {std_ms:.3f}\n"
    print(summary.strip())
    log_f.write(summary)
    log_f.close()
    return mean_ms, std_ms



    print(f"[{rule.upper()}] Mean makespan over {n_episodes} episodes: {mean_ms:.3f} ± {std_ms:.3f}")
    return mean_ms, std_ms


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="rl",
        choices=["rl", "baseline"],
        help="rl: evaluate trained SB3 model; baseline: heuristic dispatch (spt/fifo/edd/random)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to SB3 .zip model (required for mode=rl)",
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["dqn", "ppo", "sac"],
        help="RL algorithm used to train the model (required for mode=rl)",
    )
    parser.add_argument(
        "--rule",
        type=str,
        default="spt",
        choices=["spt", "fifo", "edd", "random"],
        help="Baseline dispatching rule (used only for mode=baseline)",
    )
    parser.add_argument(
        "--instance",
        type=str,
        default="datasets/instance_0.json",
        help="Path to JSON or .jsp instance",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes",
    )

    args = parser.parse_args()

    if args.mode == "rl":
        if not args.model_path or not args.algo:
            raise ValueError("For mode='rl', both --model_path and --algo must be provided.")
        mean, std = evaluate_model(
            args.model_path,
            args.algo,
            args.instance,
            n_episodes=args.episodes,
            render=False,
        )
        print(f"Mean makespan (RL {args.algo}): {mean:.3f} ± {std:.3f}")
    else:
        mean, std = evaluate_baseline(
            args.rule,
            args.instance,
            n_episodes=args.episodes,
            render=False,
        )
        print(f"Mean makespan (baseline {args.rule}): {mean:.3f} ± {std:.3f}")
