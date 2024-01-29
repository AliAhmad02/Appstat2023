import numpy as np
from scipy.stats import t, norm

data = np.genfromtxt(r"AppStat2023/Exam 2016/problem 5.2 data file.txt")
mean_res = np.mean(data)
mean_res_err = np.std(data, ddof=1) / np.sqrt(len(data))
rms = np.std(data)
print(f"Typical uncertainty on data {rms}")
print(f"\n Mean residual {mean_res}+/-{mean_res_err}")
t_val = (0 - mean_res)/mean_res_err
df_t_test = len(data) - 1
p_val_t = t.sf(t_val, df_t_test)
print(f"\n p-value consistency with 0, t test: {p_val_t:.3f}")

z_vals = (data - mean_res)/rms

p_vals_local = np.array([norm.sf(np.abs(val)) for val in z_vals])
p_vals_global = 1 - (1-p_vals_local)**len(data)
print(data[p_vals_global<0.01])