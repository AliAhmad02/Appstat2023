import numpy as np
import matplotlib.pyplot as plt

x_mu = 2.5
x_sigma = 1
Nx = 10_000

def get_p_mean(x_mean, x_std, Nx):
    x_vals = np.random.normal(loc=x_mean, scale=x_std, size=Nx)
    p_vals = np.abs(x_vals) / 4
    p_vals[np.abs(x_vals) >= 4] = 0
    p_mean = np.mean(p_vals)
    p_err = np.std(p_vals, ddof=1) / np.sqrt(len(x_vals))
    return p_mean, p_err

p_mean, p_err = get_p_mean(x_mu, x_sigma, Nx)
print(f"The mean probability is {p_mean:.3f} +/- {p_err:.3f}")

def get_p_mean_range(x_range, x_std, Nx):
    p_mean_range = []
    p_err_range = []
    for idx, x_val in enumerate(x_range):
        if idx % 10 == 0:
            print(f"Iteration {idx}")
        p_mean, p_err = get_p_mean(x_val, x_std, Nx)
        p_mean_range.append(p_mean)
        p_err_range.append(p_err)
    return np.array(p_mean_range), np.array(p_err_range)

x_range = np.linspace(-4, 4, 500)
p_mean_range, p_err_range = get_p_mean_range(x_range, x_sigma, Nx)
plt.figure(figsize=(7,5))
plt.errorbar(x_range, p_mean_range, p_err_range, fmt='.k', label='Average probability')
plt.xlabel("x [m]", fontsize=13)
plt.ylabel("Average probability", fontsize=15)
plt.legend(fontsize=13)
# plt.savefig(r"/home/ali/Figures/fig2.png", bbox_inches='tight', dpi=500)
plt.show()

p_highest_vals_left = []
p_highest_vals_right = []
max_p_left_list = []
max_p_left_list_err = []
max_p_right_list = []
max_p_right_list_err = []
for i in range(50):
    p_array, p_array_err = get_p_mean_range(x_range, x_sigma, Nx)
    left_mask = x_range < 0
    right_mask = x_range >= 0
    max_idx_left = np.argmax(p_array[left_mask])
    max_idx_right = np.argmax(p_array[right_mask])
    max_val_left = x_range[left_mask][max_idx_left]
    max_val_right = x_range[right_mask][max_idx_right]
    p_highest_vals_left.append(max_val_left)
    p_highest_vals_right.append(max_val_right)
    max_p_left_list.append(p_array[left_mask][max_idx_left])
    max_p_left_list_err.append(p_array_err[left_mask][max_idx_left])
    max_p_right_list.append(p_array[right_mask][max_idx_right])
    max_p_right_list_err.append(p_array_err[right_mask][max_idx_right])

p_highest_val_left = np.mean(p_highest_vals_left)
p_highest_val_left_err = np.std(p_highest_vals_left, ddof=1) / np.sqrt(len(p_highest_vals_left))
max_p_val_left = np.average(max_p_left_list, weights=1/np.array(max_p_left_list_err)**2)
max_p_val_left_err = 1 / np.sum(1 / np.array(max_p_left_list_err)**2)

p_highest_val_right = np.mean(p_highest_vals_right)
p_highest_val_right_err = np.std(p_highest_vals_right, ddof=1) / np.sqrt(len(p_highest_vals_right))
max_p_val_right = np.average(max_p_right_list, weights=1/np.array(max_p_right_list_err)**2)
max_p_val_right_err = 1 / np.sum(1 / np.array(max_p_right_list_err)**2)

p_highest_val_tot = np.average([np.abs(p_highest_val_left), p_highest_val_right], weights=1/np.array([p_highest_val_left_err, p_highest_val_right_err])**2)
p_highest_val_tot_err = 1 / np.sum(1 / np.array([p_highest_val_left_err, p_highest_val_right_err])**2)

max_p_val_tot = np.average([max_p_val_left, max_p_val_right], weights=1/np.array([max_p_val_left_err, max_p_val_right_err])**2)
max_p_val_tot_err = 1 / np.sum(1 / np.array([max_p_val_left_err, max_p_val_right_err])**2)

print(f"\n x with highest probability, left: {p_highest_val_left:.3f} +/- {p_highest_val_left_err:.3f}")
print(f"\n Probability: {max_p_val_left} +/- {max_p_val_left_err}\n")

print(f"\n x with highest probability, right: {p_highest_val_right:.3f} +/- {p_highest_val_right_err:.3f}")
print(f"\n Probability: {max_p_val_right} +/- {max_p_val_right_err}\n")

print(f"\n |x| with largest probability: {p_highest_val_tot:.5f} +/- {p_highest_val_tot_err:.5f}")
print(f"\n Probability: {max_p_val_tot} +/- {max_p_val_tot_err}\n")
