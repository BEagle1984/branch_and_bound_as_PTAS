# Bounding logic for Identical Machines Scheduling (P||Cmax)
def compute_bound(job_sizes, overhead, fixed):
    # LP relaxation: assign jobs to machines to minimize max load
    # For identical machines, assign jobs fractionally to balance loads
    n_jobs = len(job_sizes)
    n_machines = len(overhead)
    total_load = sum(job_sizes) + sum(overhead)
    LB = total_load / n_machines
    # Fractional assignment: spread jobs evenly
    X_frac = {}
    for j in range(n_jobs):
        if any(j == k for (k, _) in fixed):
            # Already fixed, skip
            continue
        for i in range(n_machines):
            X_frac[(j, i)] = 1.0 / n_machines
    # Add fixed assignments
    for (j, i) in fixed:
        X_frac[(j, i)] = 1.0
    feas = True
    return X_frac, LB, feas

