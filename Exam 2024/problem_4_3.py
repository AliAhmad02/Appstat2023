from scipy.stats import ks_2samp, t
import numpy as np

A = np.array([28.9,26.4,22.8,27.3,25.9])
B = np.array([22.4, 21.3, 25.1, 24.8, 22.5])

p_val_ks = ks_2samp(A, B).pvalue
print(f"\n Probability that A and B come from the same distribution: {p_val_ks:.3f}")
