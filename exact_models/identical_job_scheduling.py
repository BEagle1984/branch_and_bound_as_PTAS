# Exact model for Identical Machines Scheduling (P||Cmax)
# Uses brute-force for benchmarking small instances
import itertools

def exact_solver(jobs, m):
    best_makespan = float('inf')
    best_assignment = None
    for assignment in itertools.product(range(m), repeat=len(jobs)):
        machine_loads = [0] * m
        for job, machine in zip(jobs, assignment):
            machine_loads[machine] += job
        makespan = max(machine_loads)
        if makespan < best_makespan:
            best_makespan = makespan
            best_assignment = assignment
    return best_assignment, best_makespan

