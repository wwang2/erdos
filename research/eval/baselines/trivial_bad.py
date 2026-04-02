"""
Trivial bad solution: uniform step function h = 0.5 everywhere.
This gives the worst possible upper bound of 0.5.
"""

import numpy as np

n = 100
h_values = np.full(n, 0.5)
