# Main script for Identical Machines Scheduling (P||Cmax) using Branch and Bound
import numpy as np
import pandas as pd
from itertools import product
from BeB.identical_job_scheduling import BranchAndBound, round_jobs
from exact_models.identical_job_scheduling import exact_solver

job_machines_list = [(5, 2), (10, 2), (10, 5), (50, 2), (50, 5), (50, 10), (50, 15), (100, 2), (100, 5), (100, 10), (100, 15)]
node_selection_strategy_list = ["lowest_lower_bound", "depth_first", "breadth_first"]
lower_bound_list = ["lp"]
branching_rule_list = ["max_job_size", "min_job_size"]
rounding_rule_list = ["all_to_least_loaded", "best_matching"]
epsilon_list = [0.5, 0.1, 0.05, 0.01]
tests_to_do = product(epsilon_list, node_selection_strategy_list, lower_bound_list, branching_rule_list, rounding_rule_list)
tests_to_do = list(tests_to_do)
seed_min = 0
seed_max = 29
test_problem = "identical_job_scheduling"
test_type = "random_instances"
df = pd.DataFrame(columns=["seed", "n_machines", "n_jobs", "epsilon", "branching_rule", "node_selection", "rounding_rule", "lower_bound",
                           "best_solution", "best_bound", "runtime", "depth", "nodes_explored", "terminate",
                           "number_of_nodes_for_optimality", "optimal_solution", "opt_gap"])

def opt_gap(best_solution, OPT_exact, tol=1e-6):
    return abs(best_solution - OPT_exact) / max(tol, OPT_exact, best_solution)

for n_jobs, n_machines in job_machines_list:
    print(f"Starting with {n_jobs} - {n_machines}", flush=True)
    for seed in range(seed_min, seed_max + 1):
        np.random.seed(seed)
        jobs = np.random.randint(1, 100, size=n_jobs).tolist()
        OPT_exact_assignment, OPT_exact = exact_solver(jobs, n_machines)
        for epsilon, node_selection_strategy, lower_bound, branching_rule, rounding_rule in tests_to_do:
            print("Doing", epsilon, node_selection_strategy, lower_bound, branching_rule, rounding_rule)
            beb = BranchAndBound(node_selection_strategy, lower_bound, branching_rule, rounding_rule, epsilon)
            best_solution, X_int, LB, runtime, nodes_explored, nodes_opt, max_depth, terminate = beb.solve(jobs, n_machines, verbose=0, opt=OPT_exact)
            df = df._append({"seed": seed, "n_jobs": n_jobs, "n_machines": n_machines, "epsilon": epsilon,
            "branching_rule": branching_rule, "node_selection": node_selection_strategy, "rounding_rule": rounding_rule, "lower_bound": lower_bound,
            "best_solution": best_solution, "best_bound": LB, "runtime": runtime, "depth": max_depth, "nodes_explored": nodes_explored, "terminate": terminate,
            "number_of_nodes_for_optimality": nodes_opt, "optimal_solution": OPT_exact, "opt_gap": opt_gap(best_solution, OPT_exact)},
                ignore_index=True)
        print(f"Done with seed {seed}", flush=True)
        if seed % 5 == 0:
            df.to_csv(f"./output/results_{test_problem}_{test_type}.csv", index=False)
