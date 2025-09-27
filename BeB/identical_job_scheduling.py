import itertools as it
import time
from heapq import heappush, heappop
import numpy as np
from bounds.identical_job_scheduling import compute_bound
from utils import is_integer_val, is_integer_sol

def round_jobs(jobs, epsilon):
    # Round job sizes to a small number of types
    max_job = max(jobs)
    rounded_jobs = [int((j / max_job) // epsilon) * epsilon * max_job for j in jobs]
    job_types = list(set(rounded_jobs))
    return rounded_jobs, job_types

def assign_small_jobs(assignment, jobs, m):
    # Greedily assign small jobs to the least loaded machine
    machine_loads = [[] for _ in range(m)]
    for j in jobs:
        idx = min(range(m), key=lambda i: sum(machine_loads[i]))
        machine_loads[idx].append(j)
    return machine_loads

class Node:
    def __init__(self, X_frac, LB, depth, strategy, fixed, overhead):
        self.LB = LB
        self.X_frac = X_frac
        self.strategy = strategy
        self.depth = depth
        self.fixed = fixed
        self.overhead = overhead  # Vector of length n_machines
        self.UB = None
        self.X_int = None
    def __lt__(self, other):
        if self.strategy == "lowest_lower_bound":
            return self.LB <= other.LB
        elif self.strategy == "depth_first":
            return self.depth >= other.depth
        elif self.strategy == "breadth_first":
            return self.depth <= other.depth
    def update(self, X_int, UB):
        self.UB = UB
        self.X_int = X_int
    def __str__(self):
        return f"fixed = {self.fixed}"

class BranchAndBound:
    def __init__(self, node_selection_strategy, lower_bound, branching_rule, rounding_rule, epsilon):
        self.LLB = float("inf")
        self.LUB = float("-inf")
        self.node_selection_strategy = node_selection_strategy
        self.lower_bound_strategy = lower_bound
        self.branching_rule = branching_rule
        self.rounding_rule = rounding_rule
        self.epsilon = epsilon
        assert 0 < self.epsilon <= 1, "Epsilon must be between 0 (<) and 1 (= 1 == Exact B&B)"
        self.job_sizes = None
        self.n_machines = None
        self.n_jobs = None
        self.LUB_argmin = None
        self.verbose = None
        self.TOL = None
        self.MAX_NODES = None
    def lower_bound(self, job_sizes, overhead, fixed):
        # Returns (X_frac, value, solvable)
        return compute_bound(job_sizes, overhead, fixed)
    def branching_variable(self, X_frac):
        fractional_jobs = [j for (j, i) in X_frac.keys() if not is_integer_val(X_frac[(j, i)])]
        fractional_jobs = list(set(fractional_jobs))
        # For identical machines, just pick one fractional job to branch on
        if not fractional_jobs:
            return None
        if self.branching_rule == "max_job_size":
            return max(fractional_jobs, key=lambda j: self.job_sizes[j])
        if self.branching_rule == "min_job_size":
            return min(fractional_jobs, key=lambda j: self.job_sizes[j])
        # Default: pick the first
        return fractional_jobs[0]
    def rounding(self, X_frac, fixed):
        X_int = {k: v for k, v in X_frac.items() if is_integer_val(v) and k not in fixed}
        machine_loads = [0] * self.n_machines
        for (j, i) in X_int.keys():
            if abs(X_int[(j, i)] - 1) < self.TOL:
                machine_loads[i] += self.job_sizes[j]
        for (j, i) in fixed:
            machine_loads[i] += self.job_sizes[j]
        fractional_jobs = [j for (j, i) in X_frac.keys() if not is_integer_val(X_frac[(j, i)])]
        fractional_jobs = list(set(fractional_jobs))
        # For identical machines, assign all fractional jobs greedily
        if self.rounding_rule == "all_to_least_loaded":
            for j in sorted(fractional_jobs, key=lambda jj: self.job_sizes[jj], reverse=True):
                i = min(range(self.n_machines), key=lambda x: machine_loads[x])
                X_int[(j, i)] = 1
                machine_loads[i] += self.job_sizes[j]
            return X_int, max(machine_loads)
        if self.rounding_rule == "best_matching":
            # If number of fractional jobs is small, try all permutations
            if len(fractional_jobs) <= 8:
                best = None
                for p in it.permutations(range(self.n_machines), min(len(fractional_jobs), self.n_machines)):
                    temp_loads = machine_loads.copy()
                    for ind, j in enumerate(fractional_jobs):
                        i = p[ind % self.n_machines]
                        temp_loads[i] += self.job_sizes[j]
                    makespan = max(temp_loads)
                    if best is None or makespan < best[1]:
                        best = p, makespan
                p, makespan = best
                for ind, j in enumerate(fractional_jobs):
                    i = p[ind % self.n_machines]
                    X_int[(j, i)] = 1
                return X_int, makespan
            else:
                # For large numbers, fall back to greedy
                for j in sorted(fractional_jobs, key=lambda jj: self.job_sizes[jj], reverse=True):
                    i = min(range(self.n_machines), key=lambda x: machine_loads[x])
                    X_int[(j, i)] = 1
                    machine_loads[i] += self.job_sizes[j]
                return X_int, max(machine_loads)
    def stopping_criterion(self):
        return self.LUB / self.LLB < 1 + self.epsilon
    def solve(self, job_sizes, n_machines, verbose=0, opt=None):
        self.job_sizes = job_sizes
        self.n_machines = n_machines
        self.n_jobs = len(job_sizes)
        self.LLB = float("-inf")
        self.LUB = float("inf")
        self.LUB_argmin = None
        self.verbose = verbose
        self.TOL = 1e-6
        self.MAX_NODES = 1e4
        start = time.time()
        depth = 0
        X_frac, LB, feas = self.lower_bound(self.job_sizes, [0]*self.n_machines, [])
        self.LLB = LB
        if is_integer_sol(X_frac):
            self.LUB = LB
            self.LUB_argmin = X_frac
            if verbose >= 1:
                print("LB = ", LB, "UB = ", LB, "-- Solved at root node", flush=True)
            return LB, X_frac, LB, time.time() - start, 0, 0, 0, True
        X_int, UB = self.rounding(X_frac, [])
        root_node = Node(X_frac, LB, depth, self.node_selection_strategy, [], [0]*self.n_machines)
        root_node.update(X_int, UB)
        self.LUB = UB
        self.LUB_argmin = X_int
        queue = []
        heappush(queue, root_node)
        if verbose >= 0.5:
            print("Root node: UB = ", UB, "LB = ", LB, flush=True)
        nodes_explored = 1
        max_depth = 0
        nodes_opt = -1
        not_yet_opt = True
        while queue:
            parent_node = heappop(queue)
            max_depth = max(max_depth, len(parent_node.fixed))
            nodes_explored += 1
            if verbose >= 2:
                print(f"Exploring node {nodes_explored - 1}")
                print(f"Node LB: {parent_node.LB}, Node UB: {parent_node.UB}")
                print(f"path of the node: {parent_node.fixed}")
            j = self.branching_variable(parent_node.X_frac)
            for q in range(self.n_machines):
                new_overhead = parent_node.overhead.copy()
                new_overhead[q] += self.job_sizes[j]
                new_fixed = parent_node.fixed + [(j, q)]
                if len(new_fixed) == self.n_jobs:
                    UB = max(new_overhead)
                    if UB < self.LUB:
                        self.LUB = UB
                        self.LUB_argmin = {k: 1 for k in new_fixed}
                    continue
                X_frac, LB, _ = self.lower_bound(self.job_sizes, new_overhead, new_fixed)
                for (k, i) in new_fixed:
                    X_frac[(k, i)] = 1
                if LB >= self.LUB:
                    continue
                if is_integer_sol(X_frac):
                    self.LUB = LB
                    self.LUB_argmin = X_frac
                    continue
                X_int, UB = self.rounding(X_frac, new_fixed)
                for (k, i) in new_fixed:
                    X_int[(k, i)] = 1
                node = Node(X_frac, LB, parent_node.depth + 1, self.node_selection_strategy, new_fixed, new_overhead)
                node.update(X_int, UB)
                heappush(queue, node)
                if UB < self.LUB:
                    self.LUB = UB
                    self.LUB_argmin = X_int
            self.LLB = min(self.LUB, min(node.LB for node in queue) if queue else self.LUB)
            if abs(self.LUB - opt) <= self.TOL and not_yet_opt:
                nodes_opt = nodes_explored
                not_yet_opt = False
            if nodes_explored > self.MAX_NODES:
                if not_yet_opt:
                    nodes_opt = nodes_explored
                return self.LUB, self.LUB_argmin, self.LLB, time.time() - start, nodes_explored, nodes_opt, max_depth, False
            if self.stopping_criterion():
                if not_yet_opt:
                    nodes_opt = nodes_explored
                return self.LUB, self.LUB_argmin, self.LLB, time.time() - start, nodes_explored, nodes_opt, max_depth, True
        nodes_opt = nodes_explored
        return self.LUB, self.LUB_argmin, self.LLB, time.time() - start, nodes_explored, nodes_opt, max_depth, True
