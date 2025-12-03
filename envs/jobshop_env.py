# envs/jobshop_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Tuple
import copy

class Job:
    def __init__(self, job_id: int, ops: List[Tuple[int, float]]):
        """
        ops: list of (machine_id, processing_time) in order
        """
        self.job_id = job_id
        self.ops = ops
        self.next_op_idx = 0            # index of next operation to be executed
        self.completion_time = None     # time when last op finished
        self.in_process = False         # True while current operation is running

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
        self.current_job = None   # job_id currently running on this machine

    def is_idle(self, current_time: float) -> bool:
        return self.current_job is None or self.busy_until <= current_time


class JobShopEnv(gym.Env):
    """
    Discrete-event Job-Shop Scheduling environment.

    One action = dispatch one *ready* operation.
    Time is advanced internally to the next decision point (when at least one
    operation can start again).

    - Observation: flattened job + machine features
    - Action: integer index selecting which ready operation to dispatch.
      We keep a fixed discrete space of size max_ready_ops and use a mask.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, jobs: List[Job], n_machines: int,
                 max_steps: int = 1000, reward_shaping: str = "sparse"):
        super().__init__()
        self._orig_jobs = jobs
        self.n_machines = n_machines
        self.max_steps = max_steps
        self.reward_shaping = reward_shaping

        # Maximum possible number of operations = sum over all jobs
        self.max_ready_ops = sum(len(j.ops) for j in jobs)

        # Observation:
        # per job: one-hot(machine), proc_time, progress_ratio
        # per machine: time_to_free
        # + current_time scalar
        n_jobs = len(jobs)
        obs_dim = (n_jobs * (n_machines + 2)) + n_machines + 1
        self.observation_space = spaces.Box(
            low=-1e6, high=1e6, shape=(obs_dim,), dtype=np.float32
        )

        # Action: index into "ready list" (0..max_ready_ops-1), with masking
        self.action_space = spaces.Discrete(self.max_ready_ops)

        self.seed()
        self._reset_state()

    # ------------------------------------------------------------------ utils
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def _reset_state(self):
        # Deep copy jobs so that env episodes are independent
        self.jobs: List[Job] = [copy.deepcopy(j) for j in self._orig_jobs]
        for j in self.jobs:
            j.next_op_idx = 0
            j.completion_time = None
            j.in_process = False

        self.machines: List[Machine] = [Machine(i) for i in range(self.n_machines)]
        self.current_time: float = 0.0
        self.step_count: int = 0

        # Discrete-event queue: (finish_time, machine_id, job_id, op_index)
        self.events: List[Tuple[float, int, int, int]] = []

        # For Gantt plotting: (job_id, machine_id, start, finish, op_index)
        self.timeline: List[Tuple[int, int, float, float, int]] = []

        self.done: bool = False

        # Fast-forward to first decision point (if any)
        self._fast_forward_until_decision()

    # ---------------------------------------------------------------- decision logic
    def _collect_ready_operations(self):
        """
        Returns list of (job_id, machine_id, proc_time) that are READY to start
        at current_time: job not done, not currently running, and required
        machine is idle.
        """
        ready = []
        for job in self.jobs:
            if job.is_done() or job.in_process:
                continue
            op = job.next_op()
            if op is None:
                continue
            m_id, p = op
            mach = self.machines[m_id]
            if mach.is_idle(self.current_time):
                ready.append((job.job_id, m_id, p))
        return ready

    def _get_valid_action_mask(self):
        ready = self._collect_ready_operations()
        mask = np.zeros(self.max_ready_ops, dtype=np.float32)
        for i in range(len(ready)):
            mask[i] = 1.0
        return mask, ready

    def _fast_forward_until_decision(self):
        """
        Advance time by processing completion events until either:
        - at least one operation becomes dispatchable (ready), or
        - all jobs are done, or
        - no more events (should only happen if done).
        """
        while True:
            if all(j.is_done() for j in self.jobs):
                self.done = True
                return

            ready = self._collect_ready_operations()
            if ready:
                # there is at least one dispatchable op at current_time
                return

            if not self.events:
                # nothing running and nothing ready: dead state
                return

            # Process all events with the earliest finish time
            self.events.sort(key=lambda x: x[0])
            earliest_finish = self.events[0][0]
            self.current_time = earliest_finish

            # Collect all events finishing at this time (handle ties)
            to_process = []
            while self.events and abs(self.events[0][0] - earliest_finish) < 1e-9:
                to_process.append(self.events.pop(0))

            for finish_time, m_id, j_id, op_index in to_process:
                mach = self.machines[m_id]
                mach.current_job = None
                mach.busy_until = self.current_time

                job = self.jobs[j_id]
                job.in_process = False
                # Ensure next_op_idx reflects completion of this op
                if job.next_op_idx == op_index:
                    job.next_op_idx += 1
                else:
                    job.next_op_idx = max(job.next_op_idx, op_index + 1)
                if job.is_done():
                    job.completion_time = self.current_time

    # ---------------------------------------------------------------- gym API
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self._reset_state()
        obs = self._make_obs()
        return obs, {}

    def _make_obs(self) -> np.ndarray:
        feat = []
        n_machines = self.n_machines

        # Job features
        for job in self.jobs:
            if job.is_done():
                feat.extend([0.0] * n_machines)   # one-hot
                feat.append(0.0)                  # proc time
                feat.append(1.0)                  # progress ratio
            else:
                op = job.next_op()
                if op is None:
                    feat.extend([0.0] * n_machines)
                    feat.append(0.0)
                    feat.append(1.0)
                else:
                    m_id, p = op
                    onehot = [0.0] * n_machines
                    onehot[m_id] = 1.0
                    feat.extend(onehot)
                    feat.append(float(p))
                    ratio = float(job.next_op_idx) / max(1, len(job.ops))
                    feat.append(ratio)

        # Machine features: time to free (0 if idle)
        for mach in self.machines:
            ttf = max(0.0, mach.busy_until - self.current_time)
            feat.append(float(ttf))

        # Global time
        feat.append(float(self.current_time))

        return np.array(feat, dtype=np.float32)

    def step(self, action: int):
        """
        One step:
        - We assume we are at a decision point (after reset or after the
          previous call has fast-forwarded).
        - We map invalid actions to a valid ready operation (so the agent
          cannot "cheat" by picking invalid indices).
        - We dispatch that operation and then fast-forward time to the next
          decision point.
        """
        if self.done:
            # Once done, further step() calls just return terminal state.
            obs = self._make_obs()
            return obs, 0.0, True, False, {}

        self.step_count += 1
        info = {}

        # Get ready ops and mask
        mask, ready = self._get_valid_action_mask()
        valid_idxs = np.where(mask > 0.0)[0]

        if len(valid_idxs) == 0:
            # No dispatchable operation at this time: just fast-forward
            self._fast_forward_until_decision()
            obs = self._make_obs()
            done = all(j.is_done() for j in self.jobs)
            truncated = (self.step_count >= self.max_steps) and (not done)
            reward = 0.0
            return obs, float(reward), bool(done), bool(truncated), info

        # Remap invalid actions to a valid index (avoid "do-nothing" cheating)
        if action < 0 or action >= len(mask) or mask[action] == 0.0:
            action = int(self.np_random.choice(valid_idxs))

        # Dispatch the chosen ready operation
        job_id, machine_id, proc_time = ready[action]
        job = next(j for j in self.jobs if j.job_id == job_id)
        mach = self.machines[machine_id]

        # Machine should be idle here
        start_time = max(self.current_time, mach.busy_until)
        finish_time = start_time + float(proc_time)

        mach.busy_until = finish_time
        mach.current_job = job_id
        job.in_process = True

        # Operation index (before completion)
        op_index = job.next_op_idx

        # Record for Gantt and event queue
        self.timeline.append(
            (job.job_id, machine_id, float(start_time), float(finish_time), int(op_index))
        )
        self.events.append((finish_time, machine_id, job.job_id, int(op_index)))

        # Fast-forward until next decision
        self._fast_forward_until_decision()

        # Compute reward
        done = all(j.is_done() for j in self.jobs)
        if done:
            makespan = max(j.completion_time for j in self.jobs)
            reward = -float(makespan)
            self.done = True
        else:
            if self.reward_shaping == "dense":
                reward = -0.01 * float(self.current_time)
            else:
                reward = 0.0

        obs = self._make_obs()
        truncated = (self.step_count >= self.max_steps) and (not done)
        return obs, float(reward), bool(done), bool(truncated), info

    def render(self, mode="human"):
        s = f"t={self.current_time:.2f}\n"
        for j in self.jobs:
            s += (
                f"Job {j.job_id}: next_op_idx={j.next_op_idx}/{len(j.ops)} "
                f"in_process={j.in_process} completed={j.is_done()} "
                f"completion_time={j.completion_time}\n"
            )
        for m in self.machines:
            s += (
                f"Machine {m.machine_id}: busy_until={m.busy_until:.2f}, "
                f"current_job={m.current_job}\n"
            )
        print(s)

    # ---------------------------------------------------------------- best-schedule loader (for plotting only)
    def load_best_schedule(self, schedule):
        """
        Populate `timeline` and job completion times from an external schedule:
        schedule is a list where each item is either:
        - dict(job_id, machine_id, start, finish, op_index) or
        - [job_id, machine_id, start, finish, (op_index)]
        Used only for plotting benchmark Gantt charts.
        """
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

        # Update completion times and current_time based on timeline
        for j in self.jobs:
            ends = [ft for (jid, _, _, ft, _) in self.timeline if jid == j.job_id]
            if ends:
                j.completion_time = max(ends)
                j.next_op_idx = len(j.ops)
        self.current_time = max((t[3] for t in self.timeline), default=0.0)


# ---------------------------------------------------------------- helper to create random instance
def generate_random_instance(n_jobs=5, n_machines=3, max_ops=4, seed=None):
    rng = np.random.RandomState(seed)
    jobs = []
    for j in range(n_jobs):
        n_ops = rng.randint(1, max_ops + 1)
        ops = []
        for _ in range(n_ops):
            m = rng.randint(0, n_machines)
            p = float(rng.randint(1, 10))  # processing time 1..9
            ops.append((m, p))
        jobs.append(Job(j, ops))
    return jobs
