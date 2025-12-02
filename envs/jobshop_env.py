# envs/jobshop_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Dict, Tuple, Any
import copy

class Job:
    def __init__(self, job_id: int, ops: List[Tuple[int, int]]):
        """
        ops: list of (machine_id, processing_time) in order
        """
        self.job_id = job_id
        self.ops = ops
        self.next_op_idx = 0
        self.completion_time = None

    def next_op(self):
        if self.next_op_idx < len(self.ops):
            return self.ops[self.next_op_idx]
        return None

    def is_done(self):
        return self.next_op_idx >= len(self.ops)

class Machine:
    def __init__(self, machine_id: int):
        self.machine_id = machine_id
        self.busy_until = 0.0
        self.current_job = None

    def is_idle(self, current_time):
        return self.current_job is None or self.busy_until <= current_time

class JobShopEnv(gym.Env):
    """
    Simplified discrete event Job-Shop Scheduling environment (single action selects a ready operation to dispatch).
    - Observation: flattened features for each job and each machine (can be extended)
    - Action: integer index selecting which ready operation to dispatch next (variable size handled via masking)
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, jobs: List[Job], n_machines: int, max_steps: int = 1000, reward_shaping: str = "sparse"):
        super().__init__()
        self._orig_jobs = jobs
        self.n_machines = n_machines
        self.max_steps = max_steps
        self.reward_shaping = reward_shaping

        # We'll define a fixed maximum number of concurrent "ready operations" as sum of jobs
        self.max_ready_ops = sum(len(j.ops) for j in jobs)

        # Observation: we'll provide:
        # - job_next_proc_machine (one-hot over machines) flattened
        # - job_next_proc_time (scalar)
        # - job_next_op_index / total_ops ratio (scalar)
        # - machine busy until - current_time (time to free)
        # We'll normalize values to reasonable ranges.
        obs_dim = (len(jobs) * (n_machines + 2)) + (n_machines * 1) + 1  # +1 for current_time
        self.observation_space = spaces.Box(low=-1000.0, high=1000.0, shape=(obs_dim,), dtype=np.float32)

        # Action: choose an index from 0..(max_ready_ops-1); some actions invalid (masked)
        self.action_space = spaces.Discrete(self.max_ready_ops)

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _reset_state(self):
        # deep copy jobs
        self.jobs = [copy.deepcopy(j) for j in self._orig_jobs]
        self.machines = [Machine(i) for i in range(self.n_machines)]
        self.current_time = 0.0
        self.step_count = 0
        # maintain event queue as list of (finish_time,machine_id,job_id)
        self.events = []
        # record executed operations for Gantt plotting
        self.timeline = []  # list of (job_id, machine_id, start_time, finish_time, op_index)
        self.done = False

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self._reset_state()
        obs = self._make_obs()
        return obs, {}

    def _collect_ready_operations(self) -> List[Tuple[int,int,int]]:
        # return list of (job_id, machine_id, proc_time) that are ready (next op exists, machine idle or will be idle)
        ready = []
        for job in self.jobs:
            if job.is_done():
                continue
            m, p = job.next_op()
            # ready if previous op done (we ensure sequential) — since we only advance by dispatch, it's ready if next_op exists
            ready.append((job.job_id, m, p))
        return ready

    def _get_valid_action_mask(self):
        ready = self._collect_ready_operations()
        mask = np.zeros(self.max_ready_ops, dtype=np.float32)
        for i in range(len(ready)):
            mask[i] = 1.0
        return mask, ready

    def _make_obs(self) -> np.ndarray:
        # build feature vector
        feat = []
        ready = self._collect_ready_operations()
        # For each job, job features
        for job in self.jobs:
            if job.is_done():
                # zeros
                feat.extend([0.0] * self.n_machines)
                feat.append(0.0)  # proc time
                feat.append(1.0)  # done ratio
            else:
                m, p = job.next_op()
                onehot = [0.0]*self.n_machines
                onehot[m] = 1.0
                feat.extend(onehot)
                feat.append(float(p))
                ratio = float(job.next_op_idx) / max(1, len(job.ops))
                feat.append(ratio)
        # machine features (time to free)
        for mach in self.machines:
            ttf = max(0.0, mach.busy_until - self.current_time)
            feat.append(float(ttf))
        feat.append(float(self.current_time))
        return np.array(feat, dtype=np.float32)

    def step(self, action: int):
        """
        Action semantics: choose the i-th ready operation in the ready list (indexed 0..len(ready)-1)
        If action invalid (masked), we treat as no-op and give big negative reward to discourage.
        After dispatching we schedule the operation and advance time to next completion event (discrete event sim).
        """
        mask, ready = self._get_valid_action_mask()
        info = {}
        self.step_count += 1

        if action < 0 or action >= len(mask) or mask[action] == 0.0:
            # invalid action
            reward = -1.0
            # don't change world but penalize and step time a bit
            self.current_time += 0.1
            obs = self._make_obs()
            done = False
            if self.step_count >= self.max_steps:
                done = True
            return obs, reward, done, False, info

        # dispatch chosen ready operation
        job_id, machine_id, proc_time = ready[action]
        job = next(j for j in self.jobs if j.job_id == job_id)
        mach = self.machines[machine_id]

        # if machine busy until later, we wait until it's free (we could support queuing but here we start at available time)
        start_time = max(self.current_time, mach.busy_until)
        finish_time = start_time + proc_time
        mach.busy_until = finish_time
        mach.current_job = job_id

        # mark job's op progressed at finish time -> so we add an event
        self.events.append((finish_time, machine_id, job_id))
        # log timeline entry using current job's operation index before increment
        self.timeline.append((job.job_id, machine_id, float(start_time), float(finish_time), int(job.next_op_idx)))

        # advance current_time to earliest event (discrete event simulation)
        self.events.sort(key=lambda x: x[0])
        next_event = self.events.pop(0)
        ev_time, ev_machine, ev_job = next_event
        # advance to event time
        self.current_time = ev_time
        # finish the op
        # update machine free
        self.machines[ev_machine].current_job = None
        self.machines[ev_machine].busy_until = self.current_time
        # update job
        job_obj = next(j for j in self.jobs if j.job_id == ev_job)
        job_obj.next_op_idx += 1
        if job_obj.is_done():
            job_obj.completion_time = self.current_time

        # reward shaping
        done = all(j.is_done() for j in self.jobs)
        if done:
            makespan = max(j.completion_time for j in self.jobs)
            reward = -makespan  # we want to minimize makespan
            self.done = True
        else:
            if self.reward_shaping == "sparse":
                reward = 0.0
            elif self.reward_shaping == "dense":
                # small negative per time to encourage finishing sooner
                reward = -0.01 * (self.current_time)
            else:
                reward = 0.0

        obs = self._make_obs()
        truncated = self.step_count >= self.max_steps
        return obs, float(reward), bool(done), bool(truncated), info

    def render(self, mode="human"):
        s = f"t={self.current_time:.2f}\n"
        for j in self.jobs:
            s += f"Job {j.job_id}: next_op_idx={j.next_op_idx}/{len(j.ops)} completed={j.is_done()}\n"
        for m in self.machines:
            s += f"Machine {m.machine_id}: busy_until={m.busy_until:.2f}\n"
        print(s)

    def load_best_schedule(self, schedule):
        self.timeline = []
        for item in schedule:
            if isinstance(item, dict):
                job_id = int(item.get("job_id", 0))
                machine_id = int(item.get("machine_id", 0))
                start_time = float(item.get("start", 0.0))
                finish_time = float(item.get("finish", item.get("end", start_time)))
                op_index = int(item.get("op_index", 0))
            else:
                vals = list(item)
                job_id = int(vals[0])
                machine_id = int(vals[1])
                start_time = float(vals[2])
                finish_time = float(vals[3])
                op_index = int(vals[4]) if len(vals) >= 5 else 0
            self.timeline.append((job_id, machine_id, start_time, finish_time, op_index))
        for j in self.jobs:
            ends = [ft for (jid, _, _, ft, _) in self.timeline if jid == j.job_id]
            if ends:
                j.completion_time = max(ends)
        self.current_time = max([t[3] for t in self.timeline]) if self.timeline else 0.0

# helper to create random instance
def generate_random_instance(n_jobs=5, n_machines=3, max_ops=4, seed=None):
    rng = np.random.RandomState(seed)
    jobs = []
    for j in range(n_jobs):
        n_ops = rng.randint(1, max_ops+1)
        ops = []
        for _ in range(n_ops):
            m = rng.randint(0, n_machines)
            p = float(rng.randint(1, 10))  # processing time 1..9
            ops.append((m, p))
        jobs.append(Job(j, ops))
    return jobs
