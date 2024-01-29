import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from scipy.stats import chi2

n_containers = 100_000
mu_truck = 0
sigma_truck = 120
mu_ship = 130
sigma_ship = 50

frac_late_vals = []
for i in range(100):
    truck_arrivals = np.random.normal(loc=mu_truck, scale=sigma_truck, size=n_containers)
    ship_departures = np.random.normal(loc=mu_ship, scale=sigma_ship, size=n_containers)
    frac_late = (truck_arrivals > np.max(ship_departures)).sum() / n_containers
    frac_late_vals.append(frac_late)
print(f"\n Fraction of containers arriving too late {np.mean(frac_late_vals)} +/- {np.std(frac_late_vals)/np.sqrt(len(frac_late_vals))}")

def calculate_waiting_times(trucks, ships):
    wait_times = np.array([])
    for truck in trucks:
        ships_after_truck = ships[ships>=truck]
        if ships_after_truck.size > 0:
            wait_time = np.min(ships_after_truck) - truck
        else:
            wait_time = 24 * 60
        wait_times = np.append(wait_times,wait_time)
    return wait_times

def find_minimum_dt():
    n_containers = 1000
    mu_truck = 0
    sigma_truck = 120
    sigma_ship = 50
    mu_ship_array = np.linspace(10, 400, 1000)
    mean_waits = []
    for index, mu_ship in enumerate(mu_ship_array):
        if index % 50 == 0:
            print(f"Iteration {index}")
        truck_arrivals = np.random.normal(loc=mu_truck, scale=sigma_truck, size=n_containers)
        ship_departures = np.random.normal(loc=mu_ship, scale=sigma_ship, size=n_containers)
        wait_times = calculate_waiting_times(truck_arrivals, ship_departures)
        mean_waits.append(np.mean(wait_times))
    return mu_ship_array, mean_waits

mu_array, mean_waits = find_minimum_dt()
min_idx = np.argmin(mean_waits)

print(f"\n Minimum average wait time {mean_waits[min_idx]}")
print(f"\n Delta t for minimum average wait time {mu_array[min_idx]}")
plt.figure(figsize=(8,5))
plt.plot(mu_array, mean_waits, color='black')
plt.xlabel(r"$\Delta t$ [min]", fontsize=15)
plt.axvline(mu_array[min_idx], lw=2, color='red', linestyle='--')
plt.ylabel(r"Average waiting time [min]", fontsize=15)
plt.show()

xmin, xmax = (0, 8)
n_bins = 50
binwidth = (xmax - xmin) / n_bins

def integrated_inverse_rayleigh(x, sigma):
    return np.sqrt(2*sigma**2*np.log(1/(1-x)))

N_rayleigh = 1000
sigma_rayleigh = 2
uniform_rayleigh = np.random.uniform(size=N_rayleigh)
int_inv_ray = integrated_inverse_rayleigh(uniform_rayleigh, sigma_rayleigh)
counts, bin_edges = np.histogram(int_inv_ray, bins=n_bins, range=(xmin, xmax))
x = (bin_edges[1:] + bin_edges[:-1])/2
y = counts
mask = y > 0
x = x[mask]
y = y[mask]
sy = np.sqrt(y)

def rayleigh(x, sigma):
    return x / sigma**2 * np.exp(-1/2 * x**2 / sigma**2)

def rayleigh_fit(x, N, sigma):
    return N * binwidth * rayleigh(x, sigma)

def chi2_rayleigh(N, sigma):
    y_fit = rayleigh_fit(x, N, sigma)
    chi2 = np.sum(((y - y_fit) / sy)**2)
    return chi2

rayleigh_linspace = np.linspace(0, 8, 10_000)
minuit_rayleigh = Minuit(chi2_rayleigh, N=N_rayleigh, sigma=sigma_rayleigh)
minuit_rayleigh.errordef = 1.0
minuit_rayleigh.migrad()
N_fit, sigma_fit = minuit_rayleigh.values[:]
N_err, sigma_err = minuit_rayleigh.errors[:]
chi2_ray = minuit_rayleigh.fval
ndof_ray = len(y) - minuit_rayleigh.nfit
p_val_ray = chi2.sf(chi2_ray, ndof_ray)

fig, ax = plt.subplots(1, 1, figsize=(8,5))
ax.hist(int_inv_ray, histtype='step', bins=n_bins, range=(xmin, xmax))
ax.plot(rayleigh_linspace, rayleigh_fit(rayleigh_linspace, N_fit, sigma_fit))
ax.set_xlabel(r"$F^{-1}(r)$", fontsize=15)
ax.set_ylabel(r"Frequency", fontsize=15)
ax.text(0.95, 0.95, f"$N≈${N_fit:.2f}$\pm${N_err:.2f}\n$\sigma≈${sigma_fit:.2f}$\pm${sigma_err:.2f}\n$\chi²$≈{chi2_ray:.2f}\n$p(\chi²)$≈{p_val_ray:.2f}", transform=ax.transAxes, ma='left', va='top', ha='right', fontsize=13, family='monospace')
plt.show()
print(f"\n Best fit value for sigma {sigma_fit} +/- {sigma_err}")
print(f"\n Best fit value for N {N_fit} +/- {N_err}")
print(f"\n p-value for fit {p_val_ray}")
N_array_ray = np.arange(50, 5000)

sigma_errors = []
for N_rayleigh in N_array_ray:
    uniform_rayleigh = np.random.uniform(size=N_rayleigh)
    int_inv_ray = integrated_inverse_rayleigh(uniform_rayleigh, sigma_rayleigh)
    counts, bin_edges = np.histogram(int_inv_ray, bins=n_bins, range=(xmin, xmax))
    x = (bin_edges[1:] + bin_edges[:-1])/2
    y = counts
    mask = y > 0
    x = x[mask]
    y = y[mask]
    sy = np.sqrt(y)
    minuit_rayleigh = Minuit(chi2_rayleigh, N=N_rayleigh, sigma=sigma_rayleigh)
    minuit_rayleigh.errordef = 1.0
    minuit_rayleigh.migrad()
    sigma_errors.append(minuit_rayleigh.errors["sigma"])
inverse_sqrt_N = np.append(sigma_errors[0], 1/np.sqrt(N_array_ray[1:])*sigma_errors[1:])
plt.figure(figsize=(8,5))
plt.plot(N_array_ray, sigma_errors, label=r"Error on $\sigma$")
plt.plot(N_array_ray, inverse_sqrt_N, label=r"$\frac{1}{\sqrt{N}} \times \sigma(N=50)$")
plt.xlabel("N", fontsize=15)
plt.ylabel("Error on $\sigma$ from fit", fontsize=15)
plt.legend(fontsize=15, frameon=False)
plt.savefig('/home/ali/Figures/fig4.png', bbox_inches='tight', dpi=500)
plt.show()
