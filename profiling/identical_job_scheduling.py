# Profiling functions for identical job scheduling

class Profiling:
    """
    Encapsulates all profiling logic for identical job scheduling.
    Initialized with epsilon and n (number of jobs), computes bins once, and provides methods for profile key computation and similarity detection.
    Maintains an internal dictionary to track seen profile keys partitioned by depth, enabling efficient pruning of epsilon-equivalent nodes at each depth.
    """
    def __init__(self, epsilon, n):
        self.epsilon = epsilon
        self.n = n
        self.bins = self._geometric_bins(epsilon)
        # Dictionary mapping depth -> set of profile keys (histograms) seen at that depth
        self.seen_profiles_by_depth = dict()

    def profile_and_compare(self, node):
        """
        Compute the profile key (histogram) for the node and set the has_similar_profile flag.
        The flag is True if any node at the same depth has the same profile key (i.e., is epsilon-equivalent).
        This enables pruning and prioritization of similar nodes in the B&B tree.
        The internal dictionary ensures efficient lookup and avoids missing previously seen profiles.
        """
        node.profile_key = self.compute_profile_key(node)
        depth = node.depth
        if depth not in self.seen_profiles_by_depth:
            self.seen_profiles_by_depth[depth] = set()

        if node.profile_key in self.seen_profiles_by_depth[depth]:
            node.has_similar_profile = True
        else:
            node.has_similar_profile = False
            self.seen_profiles_by_depth[depth].add(node.profile_key)

    def compute_profile_key(self, node):
        """
        Compute the profile key for a node: the histogram of binned loads.
        This key is used to detect epsilon-equivalent nodes for pruning and prioritization.
        Args:
            node: Node object with 'overhead' (list of loads) and 'depth' attributes.
        Returns:
            tuple: histogram of binned loads
        """
        hist = [0] * (len(self.bins))
        for load in node.overhead:
            idx = self._bin_index(load, self.bins)
            hist[idx] += 1
        return tuple(hist)

    @staticmethod
    def _linear_bins(epsilon, n):
        """
        Generate linear bins for discretizing machine loads: 0, epsilon/n, 2*epsilon/n, ..., up to 2*(1+epsilon)^2.
        """
        hi = 2 * (1 + epsilon) ** 2
        step = epsilon / n
        bins = [0.0]
        x = step
        while x <= hi + 1e-12:
            bins.append(x)
            x += step
        return bins

    @staticmethod
    def _geometric_bins(epsilon):
        """
        Generate geometric bins for discretizing machine loads, as required by the PTAS for identical machines.
        The bins are spaced geometrically according to epsilon, covering [0, 2*(1+epsilon)^2].
        """
        hi = 2 * (1 + epsilon) ** 2
        bins = [0.0]
        x = epsilon
        while x <= hi + 1e-12:
            bins.append(x)
            x *= (1 + epsilon)
        return bins

    @staticmethod
    def _bin_index(x, bins):
        """
        Find the index of the largest bin in 'bins' that is less than or equal to x.
        Used to assign a load to its bin for profile key computation. Works for both linear and geometric bins.
        """
        lo, hi = 0, len(bins) - 1
        ans = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if bins[mid] <= x:
                ans = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return ans
