# utils/dataset.py
import json
from envs.jobshop_env import generate_random_instance, Job
from typing import List
import os

def generate_and_save(path="datasets", n_instances=50, n_jobs=5, n_machines=3, max_ops=4):
    os.makedirs(path, exist_ok=True)
    meta = []
    for i in range(n_instances):
        jobs = generate_random_instance(n_jobs=n_jobs, n_machines=n_machines, max_ops=max_ops, seed=i)
        # serialize
        jobs_ser = []
        for j in jobs:
            jobs_ser.append({"job_id": j.job_id, "ops": j.ops})
        fname = os.path.join(path, f"instance_{i}.json")
        with open(fname, "w") as f:
            json.dump({"jobs": jobs_ser, "n_machines": n_machines}, f, indent=2)
        meta.append(fname)
    print(f"Saved {n_instances} instances to {path}")
    return meta

def load_instance(fname):
    import json
    from envs.jobshop_env import Job
    with open(fname, "r") as f:
        data = json.load(f)
    jobs = [Job(int(j["job_id"]), [(int(m), float(p)) for (m,p) in j["ops"]]) for j in data["jobs"]]
    return jobs, data["n_machines"]

def load_jsp_instance(path: str):
    from envs.jobshop_env import Job
    jobs = []
    with open(path, "r") as f:
        first = f.readline().strip()
        parts = [int(x) for x in first.split()]
        if len(parts) < 2:
            raise ValueError("Invalid JSP header")
        n_jobs, n_machines = parts[0], parts[1]
        for j in range(n_jobs):
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading JSP body")
            toks = [int(x) for x in line.strip().split()]
            ops = []
            # parse pairs: (machine, proc_time), take first n_machines pairs
            pair_count = min(n_machines, len(toks)//2)
            for k in range(pair_count):
                m = int(toks[2*k])
                p = float(toks[2*k+1])
                ops.append((m, p))
            jobs.append(Job(j, ops))
        second = f.readline().strip()
        parts = [int(x) for x in second.split()]
        if len(parts) < 1:
            raise ValueError("Invalid JSP header for makespan")
        best_makespan = parts[0]
        best_schedule = []
        # The best schedule section contains n_machines lines; each line lists
        # global operation indices executed on that machine in order
        for m in range(n_machines):
            line = f.readline()
            if not line:
                raise ValueError("Unexpected EOF while reading JSP best schedule")
            toks = [int(x) for x in line.strip().split()]
            best_schedule.append(toks)
    return jobs, n_machines, best_makespan, best_schedule

if __name__ == "__main__":
    jobs, n_machines, best_makespan, best_schedule = load_jsp_instance("dataset5k/10x10_0.jsp")
    print(jobs)
    print(n_machines)
    print(best_makespan)
    print(best_schedule)
