from multiprocessing.pool import Pool
import multiprocessing
import os
import time
import numpy as np
import concurrent.futures

from exact_models.identical_job_scheduling import solve_identical_job_scheduling

class InstanceHandler:
    def __init__(self, path: str) -> None:
        self.filename_template = "Instance{seed}{sort_suffix}_J_M_-_{n_jobs}_{n_machines}_.txt"
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    def fetch(self, n_jobs: int, n_machines: int, seed: int | None = None, verbose: bool = False, sort_descending: bool = False) -> tuple[int, int, list[int], float]:
        """
        Fetch an instance with the specified number of jobs and machines.
        If the instance exists, load it from file. Otherwise, create a new instance.
        
        Args:
            n_jobs (int): Number of jobs
            n_machines (int): Number of machines
            seed (int, optional): Random seed for instance generation
            verbose (bool, optional): Enable verbose output
            sort_descending (bool, optional): Sort processing times in descending order
            
        Returns:
            tuple: (n_jobs, n_machines, processing_times, OPT_exact)
        """
        if verbose:
            print(f"Fetching instance: {n_jobs} jobs, {n_machines} machines, sorted: {sort_descending}")
        
        if self._exists(n_jobs, n_machines, seed, verbose, sort_descending):
            if verbose:
                print("Instance found, loading from file")
            return self._load(n_jobs, n_machines, seed, verbose, sort_descending)
        else:
            if verbose:
                print("Instance not found, generating new instance")
            return self._generate_and_save(n_jobs, n_machines, seed, verbose, sort_descending)
    
    def _exists(self, n_jobs: int, n_machines: int, seed: int | None = None, verbose: bool = False, sort_descending: bool = False) -> bool:
        """Check if an instance file exists for the given parameters."""
        seed_str = str(seed) if seed is not None else ""
        sort_suffix = "s" if sort_descending else ""
        filename = self.filename_template.format(seed=seed_str, sort_suffix=sort_suffix, n_jobs=n_jobs, n_machines=n_machines)
        full_path = os.path.join(self.path, filename)
        exists = os.path.exists(full_path)
        if verbose:
            print(f"Checking file: {filename} - {'Found' if exists else 'Not found'}")
        return exists

    def _load(self, n_jobs: int, n_machines: int, seed: int | None = None, verbose: bool = False, sort_descending: bool = False) -> tuple[int, int, list[int], float]:
        """Load an existing instance from file."""
        seed_str = str(seed) if seed is not None else ""
        sort_suffix = "s" if sort_descending else ""
        filename = self.filename_template.format(seed=seed_str, sort_suffix=sort_suffix, n_jobs=n_jobs, n_machines=n_machines)
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
    
    def _generate_and_save(self, n_jobs: int, n_machines: int, seed: int | None = None, verbose: bool = False, sort_descending: bool = False) -> tuple[int, int, list[int], float]:
        """Generate a new instance and save it to file."""
        processing_times, OPT_exact = self._generate(n_jobs, n_machines, seed, verbose, sort_descending)
        self._save(n_jobs, n_machines, processing_times, OPT_exact, seed, verbose, sort_descending)
        return n_jobs, n_machines, processing_times, OPT_exact
    
    def _generate(self, n_jobs: int, n_machines: int, seed: int | None = None, verbose: bool = False, sort_descending: bool = False) -> tuple[list[int], float]:
        """Generate a new instance with random processing times."""
        if verbose:
            print(f"Generating instance with seed: {seed}, sorted: {sort_descending}")
        if seed is not None:
            np.random.seed(seed)
        processing_times = np.random.randint(1, 100, n_jobs).tolist()
        
        # Sort processing times in descending order if requested
        if sort_descending:
            processing_times.sort(reverse=True)
            if verbose:
                print("Processing times sorted in descending order")
        
        if verbose:
            print("Solving for optimal makespan...")
        OPT_exact, _, status, runtime = solve_identical_job_scheduling(n_jobs, n_machines, processing_times)
        if verbose:
            print(f"Generated instance with optimal makespan: {OPT_exact}")
        return processing_times, OPT_exact
    
    def _save(self, n_jobs: int, n_machines: int, processing_times: list[int], OPT_exact: float, seed: int | None = None, verbose: bool = False, sort_descending: bool = False) -> None:
        """Save an instance to file."""
        seed_str = str(seed) if seed is not None else ""
        sort_suffix = "s" if sort_descending else ""
        filename = self.filename_template.format(seed=seed_str, sort_suffix=sort_suffix, n_jobs=n_jobs, n_machines=n_machines)
        full_path = os.path.join(self.path, filename)
        if verbose:
            print(f"Saving instance to: {filename}")
        with open(full_path, 'w') as f:
            f.write(f"Jobs, Machines = {n_jobs}, {n_machines}\n")
            # Write each processing time on a separate line
            sorted_note = " (sorted in descending order)" if sort_descending else ""
            f.write(f"Processing times{sorted_note} (line n contains processing time for job n for machines 1 to m):\n")
            for time in processing_times:
                f.write(f"{time}\n")
            f.write(f"Optimal makespan = {OPT_exact}\n")
        print(f"Instance saved to {full_path}")



if __name__ == "__main__":
    path_instances = "instances/identical_job_scheduling/"
    instance_handler = InstanceHandler(path_instances)

    n_processes = 14
    time_limit_in_seconds = 60*60  # 60 minutes per batch
    
    # Number of results to wait for before terminating remaining processes
    # If None, waits for all processes to complete
    n_results = 5

    seed = 42
    n_jobs_list = [1000]
    n_machines_list = [200, 100, 50]

    # Option to sort generated processing times in descending order (bigger to smaller)
    sort_descending = False

    instances = [(j, m) for j in n_jobs_list for m in n_machines_list if j > m]
    print(f"Processing {len(instances)} instances in parallel batches...")
    print(f"Each batch will run {n_processes} seeds in parallel for one instance")
    
    if n_results is not None:
        print(f"Early termination enabled: will stop after {n_results} successful results per batch")
    else:
        print("Early termination disabled: will wait for all results")

    total_completed = 0
    total_failed = 0

    # Process each instance as a separate batch
    for batch_idx, (n_jobs, n_machines) in enumerate(instances, 1):
        print(f"\n--- Batch {batch_idx}/{len(instances)}: Processing instance ({n_jobs} jobs, {n_machines} machines) ---")
        
        # Create process pool for this batch
        processes_pool = Pool(processes=n_processes)
        
        # Build argument list for current instance with all seeds
        batch_args = [(n_jobs, n_machines, s, False, sort_descending) for s in range(seed, seed + n_processes)]
        sort_info = " (sorted)" if sort_descending else ""
        print(f"[{time.strftime('%H:%M:%S')}] Submitting {len(batch_args)} jobs for instance ({n_jobs}, {n_machines}){sort_info}...")
        
        # Submit individual jobs with callbacks for real-time feedback
        async_results = []
        
        def create_job_callbacks(seed, job_index):
            """Create callback functions that know their specific seed and job index"""
            def job_completed_callback(result):
                n_jobs_ret, n_machines_ret, processing_times, OPT_exact = result
                print(f"  ✓ Process {job_index}/{len(batch_args)} completed - Seed {seed}: OPT={OPT_exact}")
            
            def job_error_callback(error):
                print(f"  ✗ Process {job_index}/{len(batch_args)} failed - Seed {seed}: Error={error}")
            
            return job_completed_callback, job_error_callback
        
        # Submit each job individually with callbacks
        for i, args in enumerate(batch_args):
            seed = args[2]  # Extract seed from args (n_jobs, n_machines, seed, verbose, sort_descending)
            success_callback, error_callback = create_job_callbacks(seed, i)
            
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
    print(f"Total instances processed: {len(instances)}")
    print(f"Total tasks completed: {total_completed}")
    print(f"Total tasks failed: {total_failed}")
    print(f"Overall success rate: {total_completed/(total_completed + total_failed)*100:.1f}%" if (total_completed + total_failed) > 0 else "No tasks executed")