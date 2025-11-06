from multiprocessing.pool import Pool
import multiprocessing
import os
import time
import numpy as np
import concurrent.futures

from exact_models.identical_job_scheduling import solve_identical_job_scheduling


class InstanceTemplate:
    def __init__(self, n_jobs: int, n_machines: int) -> None:
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.processing_times = []
        self.makespan = 0.0

        self._filename_prefix = "Instance"
        self._filename_infix = ""
        self._filename_suffix = "J{n_jobs}_M{n_machines}.txt"

    def get(self) -> tuple[int, int, list[int], float]:
        return self.n_jobs, self.n_machines, self.processing_times, self.makespan
    
    def set(self, n_jobs: int, n_machines: int, processing_times: list[int], makespan: float) -> None:
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.processing_times = processing_times
        self.makespan = makespan

    def filename_template(self) -> str:
        return f"{self._filename_prefix}{'_' if self._filename_infix else ''}{self._filename_infix}_{self._filename_suffix}"

    def filename(self) -> str:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _generate(self) -> None:
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def solve(self) -> tuple[int, int, list[int], float]:
        if self.processing_times is None or not self.processing_times:
            self._generate()
        self.makespan, _, _, _ = solve_identical_job_scheduling(self.n_jobs, self.n_machines, self.processing_times)
        return self.get()

    def __str__(self):
        return self.filename().replace(self._filename_prefix, "").replace(".txt", "")[1:]

class UniformInstance(InstanceTemplate):
    def __init__(self, n_jobs: int, n_machines: int, lb: int = 1, ub: int = 99, seed: int | None = None) -> None:
        """
        Initialize a uniform instance generator.

        Args:
            n_jobs (int): Number of jobs
            n_machines (int): Number of machines
            lb (int): Lower bound for processing times (inclusive)
            ub (int): Upper bound for processing times (inclusive)
            seed (int, optional): Random seed for reproducibility
        """
        super().__init__(n_jobs, n_machines)
        self._filename_infix = f"Uniform{lb}-{ub}_Seed{seed if seed is not None else 'D'}"
        self.lb = lb
        self.ub = ub
        self.seed = seed

    def filename(self) -> str:
        return self.filename_template().format(lb=self.lb, ub=self.ub, seed=self.seed, n_jobs=self.n_jobs, n_machines=self.n_machines)

    def _generate(self) -> None:
        if self.seed is not None:
            np.random.seed(self.seed)
        self.processing_times = np.random.randint(self.lb, self.ub+1, self.n_jobs).tolist()

class SmallBigInstance(InstanceTemplate):
    def __init__(self, n_sj: int, lb_sj: int, ub_sj: int, n_bj: int, lb_bj: int, ub_bj: int, n_machines: int, seed: int | None = None) -> None:
        """
        Initialize a small-big instance generator.

        Args:
            n_sj (int): Number of small jobs
            lb_sj (int): Lower bound for small job processing times (inclusive)
            ub_sj (int): Upper bound for small job processing times (inclusive)
            n_bj (int): Number of big jobs
            lb_bj (int): Lower bound for big job processing times (inclusive)
            ub_bj (int): Upper bound for big job processing times (inclusive)
            n_machines (int): Number of machines
        """
        super().__init__(n_sj + n_bj, n_machines)
        self._filename_infix = f"{n_sj}Small{lb_sj}-{ub_sj}_{n_bj}Large{lb_bj}-{ub_bj}_Seed{seed if seed is not None else 'D'}"
        self.n_sj = n_sj
        self.n_bj = n_bj
        self.lb_sj = lb_sj
        self.ub_sj = ub_sj
        self.lb_bj = lb_bj
        self.ub_bj = ub_bj
        self.seed = seed

    def filename(self) -> str:
        return self.filename_template().format(n_sj=self.n_sj, lb_sj=self.lb_sj, ub_sj=self.ub_sj, n_bj=self.n_bj, lb_bj=self.lb_bj, ub_bj=self.ub_bj, seed=self.seed, n_jobs=self.n_jobs, n_machines=self.n_machines)

    def _generate(self) -> None:
        if self.seed is not None:
            np.random.seed(self.seed)
        array_sj = np.random.randint(self.lb_sj, self.ub_sj + 1, self.n_sj)
        array_bj = np.random.randint(self.lb_bj, self.ub_bj + 1, self.n_bj)
        data = np.concatenate([array_sj, array_bj])
        np.random.shuffle(data)
        self.processing_times = data.tolist()

class InstanceHandler:
    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    def fetch(self, instance: InstanceTemplate, verbose: bool = False) -> tuple[int, int, list[int], float]:
        """
        Fetch an instance with the specified number of jobs and machines.
        If the instance exists, load it from file. Otherwise, create a new instance.
        
        Args:
            instance (InstanceTemplate): Instance template containing n_jobs, n_machines, and optionally seed
            verbose (bool, optional): Enable verbose output
            
        Returns:
            tuple: (n_jobs, n_machines, processing_times, OPT_exact)
        """
        
        if verbose:
            print(f"Fetching instance: {instance.n_jobs} jobs, {instance.n_machines} machines")

        if self._exists(instance, verbose):
            if verbose:
                print("Instance found, loading from file")
            self._load(instance, verbose)
        else:
            if verbose:
                print("Instance not found, generating new instance")
            self._solve_and_save(instance, verbose)
        return instance.get()

    def _exists(self, instance: InstanceTemplate, verbose: bool = False) -> bool:
        """Check if an instance file exists for the given parameters."""
        full_path = os.path.join(self.path,  instance.filename())
        exists = os.path.exists(full_path)
        if verbose:
            print(f"Checking file: {instance.filename()} - {'Found' if exists else 'Not found'}")
        return exists

    def _load(self, instance: InstanceTemplate, verbose: bool = False) -> None:
        """Load an existing instance from file."""
        full_path = os.path.join(self.path, instance.filename())
        if verbose:
            print(f"Loading instance from: {full_path}")

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
        makespan = float(last_line.split("=")[1].strip())
        
        if verbose:
            print(f"Loaded instance with optimal makespan: {makespan}")

        assert n_jobs_loaded == instance.n_jobs, "Number of jobs does not match"
        assert n_machines_loaded == instance.n_machines, "Number of machines does not match"
        assert len(processing_times) == n_jobs_loaded, "Number of processing times does not match number of jobs"
        instance.set(n_jobs_loaded, n_machines_loaded, processing_times, makespan)

    def _solve_and_save(self, instance: InstanceTemplate, verbose: bool = False) -> None:
        """Solve an existing instance and save the results to file."""
        start_time = time.time()
        if verbose:
            print("Solving instance...")
        instance.solve()
        if verbose:
            print(f"Solved instance with makespan: {instance.makespan} [in {time.time() - start_time:.0f} seconds]")

        self._save(instance, verbose)
        if verbose:
            print("Instance saved successfully")

    def _save(self, instance: InstanceTemplate, verbose: bool = False) -> None:
        """Save an instance to file."""
        full_path = os.path.join(self.path, instance.filename())
        if verbose:
            print(f"Saving instance to: {full_path}")
        with open(full_path, 'w') as f:
            f.write(f"Jobs, Machines = {instance.n_jobs}, {instance.n_machines}\n")
            # Write each processing time on a separate line
            f.write(f"Processing times (line n contains processing time for job n for machines 1 to m):\n")
            for time in instance.processing_times:
                f.write(f"{time}\n")
            f.write(f"Optimal makespan = {instance.makespan}\n")


if __name__ == "__main__":
    path_instances = "instances/identical_job_scheduling/"
    instance_handler = InstanceHandler(path_instances)

    n_processes = 12
    time_limit_in_seconds = 30*60  # 30 minutes per batch
    
    # Number of results to wait for before terminating remaining processes
    # If None, waits for all processes to complete
    n_results = 5

    seed = 42
    n_jobs_list = [100]
    n_machines_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    instance_template : InstanceTemplate = lambda n_jobs, n_machines, s: UniformInstance(n_jobs, n_machines, 1, 1000, s)
    instances_config = [(j, m) for j in n_jobs_list for m in n_machines_list if j > m]
    

    print(f"Processing {len(instances_config)} instances in parallel batches...")
    print(f"Each batch will run {n_processes} seeds in parallel for one instance")
    
    if n_results is not None:
        print(f"Early termination enabled: will stop after {n_results} successful results per batch")
    else:
        print("Early termination disabled: will wait for all results")

    total_completed = 0
    total_failed = 0

    # Process each instance as a separate batch
    for batch_idx, (n_jobs, n_machines) in enumerate(instances_config, 1):
        print(f"\n--- Batch {batch_idx}/{len(instances_config)}: Processing instance ({n_jobs} jobs, {n_machines} machines) ---")

        # Create process pool for this batch
        processes_pool = Pool(processes=n_processes)
        
        # Build argument list for current instance with all seeds
        batch_args = [(instance_template(n_jobs, n_machines, s), True) for s in range(seed, seed + n_processes)]
        print(f"[{time.strftime('%H:%M:%S')}] Submitting {len(batch_args)} jobs for instance ({n_jobs}, {n_machines})...")
        
        # Submit individual jobs with callbacks for real-time feedback
        async_results = []
        
        def create_job_callbacks(filename, job_index):
            """Create callback functions that know their specific seed and job index"""
            def job_completed_callback(result):
                n_jobs_ret, n_machines_ret, processing_times, makespan = result
                print(f"  ✓ Process {job_index}/{len(batch_args)} completed - {filename}: makespan={makespan}")
            
            def job_error_callback(error):
                print(f"  ✗ Process {job_index}/{len(batch_args)} failed - {filename}: Error={error}")

            return job_completed_callback, job_error_callback
        
        # Submit each job individually with callbacks
        for i, args in enumerate(batch_args):
            instance = args[0]  # Extract instance from args (UniformInstance, verbose)
            success_callback, error_callback = create_job_callbacks(instance.filename(), i)

            async_result = processes_pool.apply_async(
                instance_handler.fetch, 
                args,
                callback=success_callback,
                error_callback=error_callback
            )
            async_results.append(async_result)
        
        # Close pool to prevent new jobs
        processes_pool.close()
        
        try:
            # Wait for all results in parallel using ThreadPoolExecutor
            batch_results = []
            
            def get_result_with_index(index_and_async_result):
                i, async_result = index_and_async_result
                try:
                    result = async_result.get(timeout=time_limit_in_seconds)
                    return (i, True, result)  # (index, success, result)
                except multiprocessing.TimeoutError:
                    print(f"  ✗ Process {i+1}/{len(batch_args)} timed out after {time_limit_in_seconds}s")
                    return (i, False, None)  # (index, failed, None)
                except Exception as e:
                    print(f"  ✗ Process {i+1}/{len(batch_args)} failed: {e}")
                    return (i, False, None)  # (index, failed, None)
            
            # Submit all .get() calls to run in parallel using threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(async_results)) as executor:
                # Submit all get operations in parallel
                future_to_index = {
                    executor.submit(get_result_with_index, (i, async_result)): i 
                    for i, async_result in enumerate(async_results)
                }
                
                # Wait for results and terminate early if n_results is specified
                results_with_indices = []
                successful_results = 0
                
                for future in concurrent.futures.as_completed(future_to_index):
                    result = future.result()
                    results_with_indices.append(result)
                    
                    # Count successful results
                    if result[1]:  # result[1] is success flag
                        successful_results += 1
                    
                    # Check if we have enough successful results and should terminate early
                    if n_results is not None and successful_results >= n_results:
                        print(f"  ✓ Reached target of {n_results} successful results, terminating remaining threads...")
                        
                        # Cancel remaining futures
                        remaining_futures = [f for f in future_to_index.keys() if not f.done()]
                        for remaining_future in remaining_futures:
                            remaining_future.cancel()
                        
                        # Wait briefly for cancelled futures to finish cleanly
                        for remaining_future in remaining_futures:
                            try:
                                remaining_future.result(timeout=1)
                            except (concurrent.futures.CancelledError, concurrent.futures.TimeoutError):
                                pass
                        
                        break
                
                # Sort by original index and extract successful results
                results_with_indices.sort(key=lambda x: x[0])  # Sort by index
                batch_results = [result for _, success, result in results_with_indices if success and result is not None]
            
            successful_jobs = len(batch_results)
            total_submitted = len(batch_args)
            total_processed = len(results_with_indices)
            failed_jobs = total_processed - successful_jobs
            
            if n_results is not None and successful_jobs >= n_results:
                print(f"✓ Batch {batch_idx} completed with early termination! {successful_jobs}/{total_submitted} tasks completed (target: {n_results}).")
            elif successful_jobs == total_submitted:
                print(f"✓ Batch {batch_idx} completed successfully! All {successful_jobs} tasks completed.")
            else:
                unprocessed_jobs = total_submitted - total_processed
                status_msg = f"⚠ Batch {batch_idx} partially completed: {successful_jobs}/{total_submitted} tasks successful"
                if failed_jobs > 0:
                    status_msg += f", {failed_jobs} failed/timed out"
                if unprocessed_jobs > 0:
                    status_msg += f", {unprocessed_jobs} terminated early"
                print(status_msg + ".")
            
            total_completed += successful_jobs
            total_failed += failed_jobs
            
        except Exception as e:
            print(f"✗ Batch {batch_idx} error: {e}")
            print("Terminating remaining processes...")
            total_failed += n_processes
        
        # Wait for any remaining processes and clean up
        processes_pool.terminate()
        processes_pool.join()
        print(f"Batch {batch_idx} finished. Moving to next batch...")
    
    print(f"\n=== Final Summary ===")
    print(f"Total instances processed: {len(instances_config)}")
    print(f"Total tasks completed: {total_completed}")
    print(f"Total tasks failed: {total_failed}")
    print(f"Overall success rate: {total_completed/(total_completed + total_failed)*100:.1f}%" if (total_completed + total_failed) > 0 else "No tasks executed")