# Utilities for Identical Machines Scheduling (P||Cmax)
def is_integer_val(val, tol=1e-6):
    return abs(val - round(val)) < tol

def is_integer_sol(X_frac, tol=1e-6):
    return all(is_integer_val(v, tol) for v in X_frac.values())
