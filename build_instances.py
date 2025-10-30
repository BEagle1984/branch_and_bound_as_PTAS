import os
import numpy as np

from exact_models.identical_job_scheduling import solve_identical_job_scheduling



class InstanceHandler:
    def __init__(self, path: str) -> None:
        self.filename_template = "Instance{seed}_J_M_-_{n_jobs}_{n_machines}_.txt"
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    def fetch(self, n_jobs: int, n_machines: int, seed: int | None = None, verbose: bool = False) -> tuple[int, int, list[int], float]:
        """
        Fetch an instance with the specified number of jobs and machines.
        If the instance exists, load it from file. Otherwise, create a new instance.
        
        Args:
            n_jobs (int): Number of jobs
            n_machines (int): Number of machines
            seed (int, optional): Random seed for instance generation
            verbose (bool, optional): Enable verbose output
            
        Returns:
            tuple: (n_jobs, n_machines, processing_times, OPT_exact)
        """
        if verbose:
            print(f"Fetching instance: {n_jobs} jobs, {n_machines} machines")
        
        if self._exists(n_jobs, n_machines, seed, verbose):
            if verbose:
                print("Instance found, loading from file")
            return self._load(n_jobs, n_machines, seed, verbose)
        else:
            if verbose:
                print("Instance not found, generating new instance")
            return self._generate_and_save(n_jobs, n_machines, seed, verbose)
    
    def _exists(self, n_jobs: int, n_machines: int, seed: int | None = None, verbose: bool = False) -> bool:
        """Check if an instance file exists for the given parameters."""
        seed_str = str(seed) if seed is not None else ""
        filename = self.filename_template.format(seed=seed_str, n_jobs=n_jobs, n_machines=n_machines)
        full_path = os.path.join(self.path, filename)
        exists = os.path.exists(full_path)
        if verbose:
            print(f"Checking file: {filename} - {'Found' if exists else 'Not found'}")
        return exists

    def _load(self, n_jobs: int, n_machines: int, seed: int | None = None, verbose: bool = False) -> tuple[int, int, list[int], float]:
        """Load an existing instance from file."""
        seed_str = str(seed) if seed is not None else ""
        filename = self.filename_template.format(seed=seed_str, n_jobs=n_jobs, n_machines=n_machines)
        full_path = os.path.join(self.path, filename)
        if verbose:
            print(f"Loading instance from: {filename}")
        
        with open(full_path, 'r') as f:
            lines = f.readlines()
        
        # Parse first line (index 0) which must have format "Jobs, Machines = n_jobs, n_machines"
        first_line = lines[0].strip()
        if "Jobs, Machines =" not in first_line:
            raise ValueError(f"Invalid file format. Expected first line to start with 'Jobs, Machines =', but got: '{first_line}'")
        
        # Parse the jobs and machines count
        parts = first_line.split("=")[1].strip().split(",")
        n_jobs_loaded = int(parts[0].strip())
        n_machines_loaded = int(parts[1].strip())
        if verbose:
            print(f"Parsed: {n_jobs_loaded} jobs, {n_machines_loaded} machines")
        
        # Verify second line (index 1) is the expected description
        second_line = lines[1].strip()
        if "Processing times" not in second_line:
            raise ValueError(f"Invalid file format. Expected second line to contain 'Processing times', but got: '{second_line}'")
        
        # Read processing times from third line onwards (indices 2 to 2 + n_jobs - 1)
        processing_times = []
        for i in range(2, 2 + n_jobs_loaded):
            processing_times.append(int(lines[i].strip()))
        
        # Read optimal makespan from last line (index 2 + n_jobs_loaded)
        last_line = lines[2 + n_jobs_loaded].strip()
        if "Optimal makespan =" not in last_line:
            raise ValueError(f"Invalid file format. Expected last line to contain 'Optimal makespan =', but got: '{last_line}'")
        OPT_exact = float(last_line.split("=")[1].strip())
        
        if verbose:
            print(f"Loaded instance with optimal makespan: {OPT_exact}")
        
        assert n_jobs_loaded == n_jobs, "Number of jobs does not match"
        assert n_machines_loaded == n_machines, "Number of machines does not match"
        return n_jobs_loaded, n_machines_loaded, processing_times, OPT_exact
    
    def _generate_and_save(self, n_jobs: int, n_machines: int, seed: int | None = None, verbose: bool = False) -> tuple[int, int, list[int], float]:
        """Generate a new instance and save it to file."""
        processing_times, OPT_exact = self._generate(n_jobs, n_machines, seed, verbose)
        self._save(n_jobs, n_machines, processing_times, OPT_exact, seed, verbose)
        return n_jobs, n_machines, processing_times, OPT_exact
    
    def _generate(self, n_jobs: int, n_machines: int, seed: int | None = None, verbose: bool = False) -> tuple[list[int], float]:
        """Generate a new instance with random processing times."""
        if verbose:
            print(f"Generating instance with seed: {seed}")
        if seed is not None:
            np.random.seed(seed)
        processing_times = np.random.randint(1, 100, n_jobs).tolist()
        OPT_exact = 0.0
        # if verbose:
        #     print("Solving for optimal makespan...")
        # OPT_exact, _, status, runtime = solve_identical_job_scheduling(n_jobs, n_machines, processing_times)
        # if verbose:
        #     print(f"Generated instance with optimal makespan: {OPT_exact}")
        return processing_times, OPT_exact
    
    def _save(self, n_jobs: int, n_machines: int, processing_times: list[int], OPT_exact: float, seed: int | None = None, verbose: bool = False) -> None:
        """Save an instance to file."""
        seed_str = str(seed) if seed is not None else ""
        filename = self.filename_template.format(seed=seed_str, n_jobs=n_jobs, n_machines=n_machines)
        full_path = os.path.join(self.path, filename)
        if verbose:
            print(f"Saving instance to: {filename}")
        with open(full_path, 'w') as f:
            f.write(f"Jobs, Machines = {n_jobs}, {n_machines}\n")
            # Write each processing time on a separate line
            f.write("Processing times (line n contains processing time for job n for machines 1 to m):\n")
            for time in processing_times:
                f.write(f"{time}\n")
            f.write(f"Optimal makespan = {OPT_exact}\n")
        print(f"Instance saved to {full_path}")



if __name__ == "__main__":
    path_instances = "instances/identical_job_scheduling/"
    instance_handler = InstanceHandler(path_instances)

    n_jobs_list = [100,200]
    n_machines_list = [50]

    print(f"Processing {len(n_jobs_list) * len(n_machines_list)} instances...")
    
    instances = [(j, m) for j in n_jobs_list for m in n_machines_list if j > m]
    for i, (n_jobs, n_machines) in enumerate(instances, 1):
        print("================================")
        print(f"[{i}/{len(instances)}] Fetching instance with {n_jobs} jobs and {n_machines} machines...")
        print("--------------------------------")
        n_jobs_ret, n_machines_ret, processing_times, OPT_exact = instance_handler.fetch(n_jobs, n_machines, seed=42, verbose=True)
        print("----INSTANCE DETAILS------------")
        print(f"Jobs: {n_jobs_ret}, Machines: {n_machines_ret}")
        print(f"Processing times: {processing_times}")
        print(f"Optimal makespan: {OPT_exact}")
    
    print(f"\nCompleted processing all {len(instances)} instances.")