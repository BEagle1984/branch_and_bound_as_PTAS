from collections import deque
from pyscipopt import Model, SCIP_PARAMSETTING
from utils import is_integer_val


def solve_greedy(n_jobs: int, n_machines: int, processing_times: list[int], overhead: list[float], fixed_jobs: list[tuple[int,int]], verbose: bool = False):
    """
        Parameters
        ----------
        n_jobs : int
            Number of jobs to be scheduled.
        n_machines : int
            Number of identical machines available.
        processing_times : list
            List of processing times for each job.
        overhead : list
            List with all completion times, one for each machine
        fixed_jobs : list
            List of tuples with the fixed items, (j, i) --> item j is fixed on machine i

        Returns
        -------
        solution : dict
            Dictionary with the solution, (j, i): f means job j is assigned to machine i with fraction f
        makespan : float
            The makespan of the solution
        is_feasible : bool
            True if the solution is feasible, False otherwise
    """
    # Pre-conditions
    assert sum(overhead) == sum(processing_times[j] for j, i in fixed_jobs), f"The overhead does not match the fixed jobs: {sum(overhead)} != {sum(processing_times[j] for j, i in fixed_jobs)}"

    # Initialize the completion times for each machine
    completion_times = overhead.copy()

    # Compute the optimal makespan (C_max)
    C_max = sum(processing_times) / n_machines

    # List of unfixed jobs
    unfixed_jobs = [j for j in range(n_jobs) if j not in [j for j, i in fixed_jobs]]

    # Assign unfixed jobs to machines using a greedy approach
    solution = {}
    machines = deque(range(n_machines))
    for j in unfixed_jobs:
        p_j = processing_times[j]
        p_j_remaining = p_j
        
        while machines and p_j_remaining > 0:
            i = machines[0]
            if completion_times[i] < C_max:
                free_processing_time = C_max - completion_times[i]
                assign_processing_time = min(p_j_remaining, free_processing_time)
                # Assign job j to machine i
                solution[j, i] = assign_processing_time / p_j
                completion_times[i] += assign_processing_time
                p_j_remaining -= assign_processing_time
            else:
                # Machine i exceeds C_max, don't use it
                machines.popleft()
    
    # Post-conditions
    assert abs(sum(solution[(j, i)] for (j, i) in solution.keys())-len(unfixed_jobs)) < 1e-9, "Some unfixed jobs have not been completely assigned, or are over-assigned."

    return solution, max(completion_times), True


def binary_search(n_jobs: int, n_machines: int, processing_times: list[int], overhead, fixed, verbose=False):
    """
        Parameters
        ----------
        n_jobs : int
            Number of jobs to be scheduled.
        n_machines : int
            Number of identical machines available.
        processing_times : list
            List of processing times for each job.
        overhead : list
            List with all the initial makespan, one for each machine
        fixed : list
            List of tuples with the fixed items, (j, i) --> item j is fixed on machine i
    """

    unfixed_jobs = [j for j in range(n_jobs) if j not in [j for j, i in fixed]]

    """
    Now we do the binary search. The possible makespans are in the interval (left,right].
    In other words, "left" is never feasible, "right" is always feasible.
    We can initialize "left" as the largest minimal processing time - 1 (it can never be a makespan),
    and "right" as the assignment where each job is allocated to the machine with smallest overhead (it's always feasible)
    """
    left = max(min(processing_times[j] + overhead[i] for i in range(n_machines)) for j in unfixed_jobs) - 1

    shortest_machine = min([i for i in range(n_machines)], key=lambda t: overhead[t])
    temp_completion_times = overhead.copy()
    temp_completion_times[shortest_machine] += sum(processing_times[j] for j in unfixed_jobs)
    right = max(temp_completion_times)
    solution = {(j, shortest_machine): 1 for j in unfixed_jobs}

    while right - left > 1:
        middle = (left + right) // 2
        model = Model("Unrelated Job Scheduling")
        model.setParam("lp/resolvealgorithm", 'p')
        model.setParam('lp/initalgorithm', 'p')  # Use primal simplex for the initial LP
        model.setParam('lp/resolvealgorithm', 'p')  # Use primal simplex for re-solves

        # Configure simple Branch-and-Bound
        model.setIntParam("presolving/maxrounds", 0)  # Disable presolve
        model.setIntParam("separating/maxrounds", 0)  # Disable cutting planes
        model.setHeuristics(SCIP_PARAMSETTING.OFF)  # Disable heuristics
        model.setIntParam("presolving/maxrestarts", 0)  # Disable restarts
        model.setIntParam("propagating/maxrounds", 0)  # Disable propagation

        if not verbose:
            model.hideOutput()
        else:
            model.setParam('display/verblevel', 5)

        # Decision variables
        x = {}
        for j in unfixed_jobs:
            for i in range(n_machines):
                if processing_times[j] <= middle:
                    x[j, i] = model.addVar(vtype="C", name=f"x({i},{j})", lb=0.0)
                else:
                    x[j, i] = model.addVar(vtype="C", name=f"x({i},{j})", lb=0.0, ub=0.0)

        # Constraint 1. You have to allocate each job
        for j in unfixed_jobs:
            model.addCons(sum(x[j, i] for i in range(n_machines)) == 1)

        # Constraint 2. The completion time on each machine must be at most C_max
        for i in range(n_machines):
            model.addCons(sum(x[j, i] * processing_times[j] for j in unfixed_jobs) <= middle - overhead[i])

        model.setObjective(x[unfixed_jobs[0], 0], "minimize")

        # Optimize the model
        model.optimize()

        # Extract the solution
        if model.getStatus() == "optimal":
            solution = {(j, i): model.getVal(x[(j, i)]) for j in unfixed_jobs for i in
                        range(n_machines) if model.getVal(x[(j, i)]) > 0}
            assert len(list(set([j for (j, i) in solution.keys() if not is_integer_val(solution[(j, i)])]))) <= n_machines, "Too many fractional jobs"
            right = middle
        else:
            left = middle

    # The optimal makespan is always "right" at the last iteration
    return solution, right, True


def linear_relaxation(processing_times, overhead, fixed, verbose=False):
    """
    Parameters
    ----------
    processing_times : list
        List of lists completion times for each job on each machine.
        p[j][i] is the completion time of job j on machine i.
    overhead : list
        List with all the initial makespan, one for each machine
    fixed : list
        List of tuples with the fixed items, (j, i) --> item j is fixed on machine i
    """
    model = Model("Unrelated Job Scheduling")

    if not verbose:
        model.hideOutput()

    n_jobs = len(processing_times)
    n_machines = len(processing_times[0])

    unfixed_jobs = [j for j in range(n_jobs) if j not in [j for j, i in fixed]]

    # Decision variables
    x = {}
    for j in unfixed_jobs:
        for i in range(n_machines):
            x[j, i] = model.addVar(vtype="C", name=f"x({i},{j})", lb=0.0)

    # Makespan
    C_max = model.addVar(vtype="C", name="C_max", lb=0.0)

    # Objective function
    model.setObjective(C_max, "minimize")

    # Constraint 1. You have to allocate each job
    for j in unfixed_jobs:
        model.addCons(sum(x[j, i] for i in range(n_machines)) == 1)

    # Constraint 2. The completion time on each machine must be at most C_max
    for i in range(n_machines):
        model.addCons(sum(x[j, i] * processing_times[j][i] for j in unfixed_jobs) <= C_max - overhead[i])

    # Set the solver method to simplex (for having a feas. sol. that's a vertex)
    model.setParam("lp/resolvealgorithm", 'p')

    # Optimize the model
    model.optimize()

    # Extract the solution
    if model.getStatus() == "optimal":
        solution = {(j, i): model.getVal(x[(j, i)]) for j in unfixed_jobs for i in
                    range(n_machines) if model.getVal(x[(j, i)]) > 0}
        makespan = model.getObjVal()
        assert len(set([j for (j, i) in solution.keys() if not is_integer_val(solution[(j, i)])])) <= n_machines, "Too many fractional jobs"
        return solution, makespan, True
    else:
        return None, None, False  # Let's keep it, but it's always feasible
