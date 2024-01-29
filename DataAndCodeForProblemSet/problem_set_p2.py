import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm, pearsonr, t

x = 1.043
x_err = 0.014

y = 0.07
y_err = 0.23

z1_x_err_term = y**2 * np.exp(-2*y) * x_err**2
z1_y_err_term = x**2 * np.exp(-2*y) * (1-y)**2 * y_err**2

z2_x_err_term = (y+1)**6 / (x-1)**4 * x_err**2
z2_y_err_term = 9 * (y+1)**4 / (x-1)**2 * y_err**2
 
print(f"\n x term contribution to z1 variance {z1_x_err_term}")
print(f"\n y term contribution to z1 variance {z1_y_err_term}")

xy_corr = 0.4
xy_covar = xy_corr * x_err * y_err
z1_corr_term = 2 * x * y * np.exp(-2*y) * (1-y) * xy_covar
z2_corr_term = -6 * (y+1)**5/(x-1)**3 * xy_covar
z1_err_corr = np.sqrt(z1_x_err_term + z1_y_err_term + z1_corr_term)
z2_err_corr = np.sqrt(z2_x_err_term + z2_y_err_term + z2_corr_term)
print(f"\n Total error on z1 (correlated) {z1_err_corr}")
print(f"\n Total error on z2 (correlated) {z2_err_corr}")

def z1(x, y):
    return x*y*np.exp(-y)

def z2(x,y):
    return (y+1)**3/(x-1)

x_array, y_array = np.random.normal(loc=1.043, scale=0.014, size=10_000), np.random.normal(loc=0.07, scale=0.23, size=10_000)

z1_array, z2_array = z1(x_array, y_array), z2(x_array, y_array)

mask = (z1_array>-2) & (z1_array<2) & (z2_array>-10) & (z2_array<90)

z1_values, z2_values = z1_array[mask], z2_array[mask]

z1z2_corr = pearsonr(z1_values, z2_values)
print(f"\n Pearson Correlation: {z1z2_corr[0]}\n Probability for non-correlation: {z1z2_corr[1]}")

fig, ax = plt.subplots(1, 1, figsize=(8,5))
ax.plot(z2_values, z1_values, ".", color='black')
ax.set_xlabel(r"$z_2$", fontsize=15)
ax.set_ylabel(r"$z_1$", fontsize=15)
ax.text(0.95, 0.05, fr"$\rho_{{z_1, z_2}} \approx {z1z2_corr[0]:.2f}$", fontsize=15, horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)
plt.show()

density = np.array([5.50, 5.61, 4.88, 5.07, 5.26])
densitity_err = np.array([0.10, 0.21, 0.15, 0.14, 0.13])

density_mean = np.average(density, weights=1/densitity_err**2)
density_mean_err = np.sqrt(1 / np.sum(1/densitity_err**2))
print(f"\n The average density is {density_mean} +/- {density_mean_err}")
chi2_density = np.sum((density - density_mean)**2 / densitity_err**2)
ndof_density = 4
prob_chi2 = chi2.sf(chi2_density, ndof_density)

best_estimate_density = np.mean(density)
best_estimate_density_err = np.std(density, ddof=1) / np.sqrt(len(density))
t_val = (5.514 - best_estimate_density)/best_estimate_density_err
df_t_test = len(density) - 1
prob_from_t = t.sf(t_val, df_t_test)
print(f"\n Density value {density_mean} +/- {density_mean_err}")
print(f"\n p-value density (not consistent with each other) {prob_chi2}")
print(f"\n Best estimate of density {best_estimate_density:.2f} +/- {best_estimate_density_err:.2f}")
print(f"\n p-value consistency with exact value (t test) {prob_from_t}")

a = 1.04
a_err = 0.27

e = 0.71
e_err = 0.12

area = np.pi * a**2 * np.sqrt(1-e**2)
area_err = np.sqrt((2*np.pi*a*np.sqrt(1-e**2) * a_err)**2 + (np.pi*a**2*e/np.sqrt(1-e**2)*e_err)**2)
print(f"\n The area is {area} +/- {area_err}")

circum_lower = 4 * a * np.sqrt(2-e**2)
circum_lower_err = np.sqrt((4 * np.sqrt(2-e**2) * a_err)**2 + (4 * a * e / np.sqrt(2-e**2) * e_err)**2)
circum_upper = np.pi * a * np.sqrt(4-2*e**2)
circum_upper_err = np.sqrt((np.pi * np.sqrt(4 - 2 * e**2) * a_err)**2 + (np.pi * a * e * 2 / np.sqrt(4 - 2 * e**2) * e_err)**2)
circum_bounds_err = np.array([circum_lower_err, circum_upper_err])
c_val = np.average([circum_lower, circum_upper], weights=1/circum_bounds_err**2)
c_val_err = 1 / np.sum(1 / circum_bounds_err**2)
print(f"\n Circumference is {c_val} +/- {c_val_err}")
print(np.mean([circum_lower, circum_upper]), 1/2*np.sqrt(circum_lower_err**2+circum_upper_err**2))
