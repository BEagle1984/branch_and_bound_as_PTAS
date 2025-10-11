# Profiling functions for identical job scheduling

class Profiling:
    """
    Encapsulates all profiling logic for identical job scheduling.
    Initialized with epsilon, computes bins once, and provides methods for profile key computation and flag setting.
    """
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.bins = self._geom_bins(epsilon)

    def profile_and_compare(self, node, queue):
        """
        Compute the profile key for the node (using its overhead and depth) and set the has_similar_profile flag.
        The flag is True if any node in the queue has the same profile key (i.e., is epsilon-equivalent).
        This enables pruning and prioritization of similar nodes in the B&B tree.
        """
        node.profile_key = self.compute_profile_key(node)
        node.has_similar_profile = any(
            hasattr(n, 'profile_key') and n.profile_key == node.profile_key for n in queue
        )

    def compute_profile_key(self, node):
        """
        Compute the profile key for a node: a tuple (depth, histogram of binned loads).
        This key is used to detect epsilon-equivalent nodes for pruning and prioritization.
        Args:
            node: Node object with 'overhead' (list of loads) and 'depth' attributes.
        Returns:
            tuple: (depth, histogram of binned loads)
        """
        hist = [0] * (len(self.bins))
        for L in node.overhead:
            idx = self._bin_index_geometric(L, self.bins)
            hist[idx] += 1
        return (node.depth, tuple(hist))

    @staticmethod
    def _geom_bins(epsilon):
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
    def _bin_index_geometric(x, bins):
        """
        Find the index of the largest bin in 'bins' that is less than or equal to x.
        Used to assign a load to its geometric bin for profile key computation.
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
