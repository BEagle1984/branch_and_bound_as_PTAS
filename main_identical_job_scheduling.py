import pandas as pd
from build_instances import InstanceHandler, InstanceTemplate, SmallBigInstance, UniformInstance
from BeB.identical_job_scheduling import BranchAndBound, ProfilingMode
from datetime import datetime
import numpy as np

def opt_gap(tol=1e-6):
    if OPT_exact is None:
        return None
    else:
        return abs(best_solution - OPT_exact) / max(tol, OPT_exact, best_solution)

def validate_solution():
    assigned_jobs = {}  # maps job -> machine
    for key, val in X_int.items():
        j, m = key

        assert 0 <= j < n_jobs, f"Job index out of range in X_int key: {j}"
        assert 0 <= m < n_machines, f"Machine index out of range in X_int key: {m}"
        assert isinstance(val, (int, float)), f"X_int[{key}] has non-numeric value {val}"
        assert np.isclose(float(val), 1.0, rtol=1e-6, atol=1e-9), f"X_int[{key}]={val} must be 1 (within tolerance)"

        # ensure each job is assigned exactly once
        assert j not in assigned_jobs, f"Job {j} assigned to multiple machines ({assigned_jobs.get(j)} and {m}) in X_int"
        assigned_jobs[j] = m

    # ensure all jobs are assigned
    missing_jobs = set(range(n_jobs)) - set(assigned_jobs.keys())
    assert not missing_jobs, f"Not all jobs assigned in X_int, missing jobs: {sorted(missing_jobs)}"

    # Recompute makespan (maximum load across machines) from processing_times and assigned_jobs
    loads = [0.0] * n_machines
    for j, m in assigned_jobs.items():
        loads[m] += float(processing_times[j])
    computed_makespan = float(max(loads)) if loads else 0.0

    # Allow a small numerical tolerance when comparing floats
    assert np.isclose(computed_makespan, float(best_solution), rtol=1e-6, atol=1e-8), f"Computed makespan {computed_makespan} does not match reported best_solution {best_solution}"

    if OPT_exact is not None:
        assert round(best_solution) >= round(OPT_exact), "Our solution cannot be better than the optimal"


# Modify this to test different instances
instances : list[InstanceTemplate] = [
    # UniformInstance(100, 20, 1, 99, 45),
    # SmallBigInstance(900, 1, 9, 100, 80, 99, 150, 63),
    UniformInstance(50, 25, 1, 1000, 42),
    UniformInstance(50, 25, 1, 1000, 43),
    UniformInstance(50, 25, 1, 1000, 44),
    UniformInstance(50, 25, 1, 1000, 45),
    UniformInstance(50, 25, 1, 1000, 46),
    UniformInstance(50, 25, 1, 1000, 47),
    UniformInstance(50, 25, 1, 1000, 48),
    UniformInstance(50, 25, 1, 1000, 49),
    UniformInstance(50, 25, 1, 1000, 50),
    UniformInstance(50, 25, 1, 1000, 51),
    UniformInstance(100, 50, 1, 1000, 42),
    UniformInstance(100, 50, 1, 1000, 43),
    UniformInstance(100, 50, 1, 1000, 44),
    UniformInstance(100, 50, 1, 1000, 45),
    UniformInstance(100, 50, 1, 1000, 46),
    UniformInstance(100, 50, 1, 1000, 47),
    UniformInstance(100, 50, 1, 1000, 48),
    UniformInstance(100, 50, 1, 1000, 49),
    UniformInstance(100, 50, 1, 1000, 50),
    UniformInstance(100, 50, 1, 1000, 51),
    UniformInstance(200, 100, 1, 1000, 42),
    UniformInstance(200, 100, 1, 1000, 43),
    UniformInstance(200, 100, 1, 1000, 44),
    UniformInstance(200, 100, 1, 1000, 45),
    UniformInstance(200, 100, 1, 1000, 46),
    UniformInstance(200, 100, 1, 1000, 47),
    UniformInstance(200, 100, 1, 1000, 48),
    UniformInstance(200, 100, 1, 1000, 49),
    UniformInstance(200, 100, 1, 1000, 50),
    UniformInstance(200, 100, 1, 1000, 51),
]

compute_exact_solutions = False

tests_list = [
    {
        "epsilon": 0.004,
        "profiling_mode": ProfilingMode.NO_PROFILING,
        "node_selection": "lowest_lower_bound",
        "lower_bound": "greedy",
        "branching_rule": "max_proc",
        "rounding_rule": "all_to_shortest",
    },
    {
        "epsilon": 0.002,
        "profiling_mode": ProfilingMode.PRUNE,
        "node_selection": "lowest_lower_bound",
        "lower_bound": "greedy",
        "branching_rule": "max_proc",
        "rounding_rule": "all_to_shortest",
    }
]

# Set up the things you want to record
test_problem = "identical_job_scheduling"
test_type = "random_instances"
timestamp = datetime.now().isoformat('#','seconds').replace(":", "")

# Instance handler
path_instances = "instances/identical_job_scheduling/"
instance_handler = InstanceHandler(path_instances)

# Create a pandas data frame to store the results
df = pd.DataFrame(columns=["instance", "n_machines", "n_jobs", "epsilon", "branching_rule", "node_selection", "profiling_mode", "rounding_rule", "lower_bound",
                           "best_solution", "best_bound", "runtime", "depth", "nodes_explored", "terminate",
                           "number_of_nodes_for_optimality", "optimal_solution", "opt_gap"])

for instance in instances:
    n_jobs, n_machines, processing_times, OPT_exact = instance_handler.fetch(instance, compute_exact_solutions, verbose=True)

    print(f"Starting with {instance}", flush=True)

    for test in tests_list:
        epsilon = test["epsilon"]
        node_selection_strategy = test["node_selection"]
        profiling_mode = test["profiling_mode"]
        lower_bound = test["lower_bound"]
        branching_rule = test["branching_rule"]
        rounding_rule = test["rounding_rule"]

        print(f"Solving with eps={epsilon}, {node_selection_strategy}, {profiling_mode.name}, {lower_bound}, "
              f"{branching_rule}, {rounding_rule}", flush=True)

        beb = BranchAndBound(node_selection_strategy, profiling_mode, lower_bound, branching_rule, rounding_rule, epsilon)

        best_solution, X_int, LB, runtime, nodes_explored, nodes_opt, max_depth, terminate = beb.solve(n_jobs, n_machines, processing_times, verbose=False, opt=OPT_exact)

        print(f"\tBest solution: {best_solution}, Nodes explored: {nodes_explored}, Runtime: {runtime:.2f}s", flush=True)

        validate_solution()

        df = df._append({"instance": str(instance), "n_jobs": n_jobs, "n_machines": n_machines, "epsilon": epsilon,
        "branching_rule": branching_rule, "node_selection": node_selection_strategy, "profiling_mode": profiling_mode.name, "rounding_rule": rounding_rule, "lower_bound": lower_bound,
        "best_solution": best_solution, "best_bound": LB, "runtime": runtime, "depth": max_depth, "nodes_explored": nodes_explored, "terminate": terminate,
        "number_of_nodes_for_optimality": nodes_opt, "optimal_solution": OPT_exact, "opt_gap": opt_gap()},
        ignore_index=True)

    # Logging
    print(f"Done with instance {instance}", flush=True)

    # Save the results
    df.to_csv(f"./output/{timestamp}_{test_problem}_{test_type}.csv", index=False)
