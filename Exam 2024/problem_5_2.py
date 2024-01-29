import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2, norm
from scipy.special import erf

data = np.genfromtxt('AppStat2023/Exam 2024/data_DecayTimes.csv')
data_mean = np.mean(data)
data_mean_err = np.std(data, ddof=1) / np.sqrt(len(data))

data_median = np.median(data)
data_median_err = np.sqrt(np.pi/2) * data_mean_err

fig, ax = plt.subplots(1, 1, figsize=(7,4))
ax.hist(data, bins=90, histtype='step')
ax.text(0.95, 0.95, f'$\mu$≈{data_mean:.3f}$\pm${data_mean_err:.3f}\n$m≈${data_median:.3f}$\pm${data_median_err:.3f}', horizontalalignment='right', ma='left', va='top', ha='right', transform=ax.transAxes, fontsize=14)
ax.set_xlabel("Decay times", fontsize=15)
# plt.savefig("/home/ali/Figures/fig9.png", bbox_inches="tight", dpi=500)
plt.show()

tmin = 0.3
tmax = 8
n_bins = 70
data_new = data[(data>=tmin) & (data<=tmax)]

counts, bin_edges = np.histogram(data_new, bins=n_bins, range=(tmin, tmax))
mask = counts > 0
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
bin_width = (tmax - tmin) / n_bins

def exp_fit(x, Nexp, tau):
    return 1 / tau * Nexp * bin_width * np.exp(-x / tau)

lstq_exp = LeastSquares(bin_centers[mask], counts[mask], np.sqrt(counts[mask]), exp_fit)
minuit_exp = Minuit(lstq_exp, Nexp=1000, tau=1)
minuit_exp.migrad()
minuit_exp.hesse()

Nexp, tau = minuit_exp.values[:]
Nexp_err, tau_err = minuit_exp.errors[:]
chi2_exp = minuit_exp.fval
ndof_exp = len(bin_centers[mask]) - minuit_exp.nfit
p_val_exp = chi2.sf(chi2_exp, ndof_exp)
exp_linspace = np.linspace(tmin, tmax, 10_000)
exp_fit_vals = exp_fit(exp_linspace, Nexp, tau)
fig, ax = plt.subplots(1, 1, figsize=(7,4))
ax.hist(data, bins=90, histtype='step', label="Histogram")
ax.plot(exp_linspace, exp_fit_vals, label="Fit")
ax.text(0.95, 0.95, f"$N_{{exp}}≈${Nexp:.3f}$\pm${Nexp_err:.3f}\n$𝜏≈${tau:.3f}$\pm${tau_err:.3f}\n$\chi²$≈{chi2_exp:.3f}\n$n_{{DOF}}$={ndof_exp}\n$p(\chi², n_{{DOF}})$≈{p_val_exp:.3f}", transform=ax.transAxes, ma='left', va='top', ha='right', fontsize=13, family='monospace')
ax.set_xlabel("Decay times", fontsize=15)
plt.legend(fontsize=13, frameon=False, loc="lower right")
# plt.savefig("/home/ali/Figures/fig10.png", bbox_inches='tight', dpi=500)
plt.show()

def gauss_no_bias(x, sigma):
    return 1 / (np.sqrt(2*np.pi)*sigma) * np.exp(-0.5 * x**2 / sigma**2)

def fit_convolve(x, Nexp, tau, sigma):
    exp_term = 1 / (2*tau) * np.exp(1/(2*tau)*(sigma**2/tau-2*x))
    erf_term = 1 - erf((sigma**2/tau-x)/(np.sqrt(2)*sigma))
    return Nexp * bin_width_full *exp_term * erf_term

tmin_full = 0
tmax_full = 16
n_bins_full = 80

counts_full, bin_edges_full = np.histogram(data, bins=n_bins_full, range=(tmin_full, tmax_full))
mask_full = counts_full > 0
bin_centers_full = (bin_edges_full[1:] + bin_edges_full[:-1]) / 2
bin_width_full = (tmax_full - tmin_full) / n_bins_full

lstsq_full = LeastSquares(bin_centers_full[mask_full], counts_full[mask_full], np.sqrt(counts_full[mask_full]), fit_convolve)
minuit_full = Minuit(lstsq_full, Nexp=Nexp, tau=tau, sigma=0.7)
minuit_full.migrad()
minuit_full.hesse()

Nexp_full, tau_full, sigma_full = minuit_full.values[:]
Nexp_full_err, tau_full_err, sigma_full_err = minuit_full.errors[:]
chi2_full = minuit_full.fval
ndof_full = len(bin_centers_full[mask_full]) - minuit_full.nfit
p_val_full = chi2.sf(chi2_full, ndof_full)
linspace_full = np.linspace(tmin_full, tmax_full, 10_000)
f_vals_full = fit_convolve(linspace_full, Nexp_full, tau_full, sigma_full)


fig, ax = plt.subplots(1, 1, figsize=(7,4))
ax.hist(data, bins=n_bins_full, range=(tmin_full, tmax_full), histtype='step', label="Histogram", lw=1.1)
ax.plot(linspace_full, f_vals_full, label="Fit", ls='--', lw=1.7)
ax.set_xlabel("Decay times", fontsize=15)
ax.text(0.95, 0.95, f"$N_{{exp}}≈${Nexp_full:.3f}$\pm${Nexp_full_err:.3f}\n$𝜏≈${tau_full:.3f}$\pm${tau_full_err:.3f}\n$\sigma≈${sigma_full:.3f}$\pm${sigma_full_err:.3f}\n$\chi²$≈{chi2_full:.3f}\n$n_{{DOF}}$={ndof_full}\n$p(\chi², n_{{DOF}})$≈{p_val_full:.3f}", transform=ax.transAxes, ma='left', va='top', ha='right', fontsize=13, family='monospace')
plt.legend(fontsize=13, frameon=False, loc="lower right")
# plt.savefig("/home/ali/Figures/fig11.png", bbox_inches='tight', dpi=500)
plt.show()

z_tau = (tau_full - tau) / np.sqrt(tau**2 + tau_full**2)
p_val_z = 2 * norm.cdf(-np.abs(z_tau))
print(f"\n Consistency check tau fit values: {p_val_z:.3f}")
