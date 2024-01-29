import numpy as np
import pandas as pd
from iminuit import Minuit
import matplotlib.pyplot as plt
from scipy.stats import chi2

dataframe = pd.read_csv('AppStat2023/DataAndCodeForProblemSet/data_GlacierSizes.csv', header=0, index_col=None)
Area, sigArea, Volume, sigVolume = dataframe.values.T

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].plot(Area, Volume, '.', color='black')
axs[0].set_xlabel(r"Area [$km²$]", fontsize=14)
axs[0].set_ylabel(r"Volume [$km³$]", fontsize=14)

axs[1].plot(sigArea / Area, '.', label=fr"$\sigma_A/A$")
axs[1].plot(sigVolume / Volume, '.', label=fr"$\sigma_V/V$")
axs[1].set_xlabel("Data point", fontsize=14)
axs[1].set_ylabel("Relative error", fontsize=14)
plt.legend(fontsize=15, frameon=False, markerscale=2)
plt.show()
print("\n Volume has the largest relative uncertainty (see plot)")

y = Volume
x = Area
sy = sigVolume
sx = sigArea

def AV_power_fit(x, c):
    return c * x**(3/2)

def chi2_AV(c):
    y_fit = AV_power_fit(x, c)
    chisq = np.sum(((y-y_fit)/sy)**2)
    return chisq

minuit_AV = Minuit(chi2_AV, c=1)
minuit_AV.errordef = 1.0
minuit_AV.migrad()

ndof_AV = len(y) - minuit_AV.nfit
c, = minuit_AV.values[:]
c_err, = minuit_AV.errors[:]
chi2_val_AV = minuit_AV.fval
p_val = chi2.sf(chi2_val_AV, ndof_AV)
x_lin = np.linspace(0, 40, 1000)
fig, axs = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[2, 1], sharex=True)
fig.subplots_adjust(hspace=0)
axs[0].errorbar(x, y, yerr=sy, fmt='.k', label='Data')
axs[0].plot(x_lin, AV_power_fit(x_lin, c), label='Fit', lw=2)
axs[0].set_ylabel("Volume $[km³]$", fontsize=15)
axs[0].text(0.95, 0.05, f"$c$≈{c:.4f}$\pm${c_err:.4f}\n$\chi²$≈{chi2_val_AV:.2f}\n$p(\chi²)$≈{p_val:.2f}", transform=axs[0].transAxes, ma='left', va='bottom', ha='right', fontsize=15, family='monospace')
axs[1].plot(x, (y-AV_power_fit(x, c))/sy, '.', color='royalblue')
axs[1].set_xlabel("Area [$km²$]", fontsize=15)
axs[1].set_ylabel("Residuals", fontsize=15)
axs[1].axhline(0, color='black', lw=1.5, ls='--')
axs[0].legend(fontsize=15, frameon=False)
# plt.savefig("/home/ali/Figures/fig11.png", bbox_inches='tight', dpi=500)
plt.show()
print(f"\n p-value for first fit. Not satisfied {p_val}")
def AV_power_fit_improved(x, b, c):
    return c * x ** b

def chi2_AV_improved(b, c):
    y_fit = AV_power_fit_improved(x, b, c)
    chisq = np.sum(((y - y_fit) / sy)**2)
    return chisq

minuit_AV_improved = Minuit(chi2_AV_improved, b=3/2, c=0.1)
minuit_AV_improved.errordef = 1.0
minuit_AV_improved.migrad()
ndof_AV_improved = len(x) - minuit_AV_improved.nfit
chi2_val_AV_improved = minuit_AV_improved.fval
b1, c1 = minuit_AV_improved.values[:]
b1_err, c1_err = minuit_AV_improved.errors[:]
p_val_AV_improved = chi2.sf(chi2_val_AV_improved, ndof_AV_improved)
print(f"\n Old chi square: {chi2_val_AV:.2f} \n New chi square: {chi2_val_AV_improved:.2f}")
dy_dx = b1 * c1 * x**(b1-1)
# dy_dx = 1
def chi2_AV_improved_with_x_errors(b, c):
    y_fit = AV_power_fit_improved(x, b, c)
    chisq = np.sum((y - y_fit)**2 / (sy**2 + sx**2 * dy_dx**2))
    return chisq

minuit_AV_improved_with_x_errors = Minuit(chi2_AV_improved_with_x_errors, b=3/2, c=0.1)
minuit_AV_improved_with_x_errors.errordef = 1.0
minuit_AV_improved_with_x_errors.migrad()
chi2_val_AV_improved_with_x_errors = minuit_AV_improved_with_x_errors.fval
b2, c2 = minuit_AV_improved_with_x_errors.values[:]
b2_err, c2_err = minuit_AV_improved_with_x_errors.errors[:]
p_val_AV_improved_with_x_errors = chi2.sf(chi2_val_AV_improved_with_x_errors, ndof_AV_improved)
print(f"\n p-value including A errors: {p_val_AV_improved_with_x_errors}")
print(f"\n chi square including A errors: {chi2_val_AV_improved_with_x_errors}")
# x_lin = np.linspace(0, 40, 1000)
# plt.figure(figsize=(6, 4))
# plt.errorbar(x, y, yerr=sy, xerr=sx, fmt='.k')
# plt.plot(x_lin, AV_power_fit_improved(x_lin, b2, c2))
# plt.show()
fig, axs = plt.subplots(2, 2, figsize=(17,6), height_ratios=[2,1], sharex='col')
fig.subplots_adjust(hspace=0)
axs[0, 0].errorbar(x, y, yerr=sy, fmt='.k', label='Data')
axs[0, 0].plot(x_lin, AV_power_fit_improved(x_lin, b1, c1), label='Fit', lw=2)
axs[0, 0].set_ylabel("Volume $[km³]$", fontsize=15)
axs[0, 0].text(0.95, 0.05, f"$c$≈{c1:.4f}$\pm${c1_err:.4f}\n$b$≈{b1:.3f}$\pm${b1_err:.3f}\n$\chi²$≈{chi2_val_AV_improved:.2f}\n$p(\chi²)$≈{p_val_AV_improved:.2f}", transform=axs[0, 0].transAxes, ma='left', va='bottom', ha='right', fontsize=15, family='monospace')
axs[1, 0].plot(x, (y-AV_power_fit_improved(x, b1, c1))/sy, '.', color='royalblue')
axs[1, 0].set_xlabel("Area [$km²$]", fontsize=15)
axs[1, 0].set_ylabel("Residuals", fontsize=15)
axs[1, 0].axhline(0, color='black', lw=1.5, ls='--')
axs[0, 0].legend(fontsize=15, frameon=False)
axs[0, 1].errorbar(x, y, yerr=sy, xerr=sx, fmt='.k', label='Data')
axs[0, 1].plot(x_lin, AV_power_fit_improved(x_lin, b2, c2), label='Fit', lw=2)
axs[0, 1].set_ylabel("Volume $[km³]$", fontsize=15)
axs[0, 1].text(0.95, 0.05, f"$c$≈{c2:.4f}$\pm${c2_err:.4f}\n$b$≈{b2:.3f}$\pm${b2_err:.3f}\n$\chi²$≈{chi2_val_AV_improved_with_x_errors:.2f}\n$p(\chi²)$≈{p_val_AV_improved_with_x_errors:.2f}", transform=axs[0, 1].transAxes, ma='left', va='bottom', ha='right', fontsize=15, family='monospace')
axs[1, 1].plot(x, (y-AV_power_fit_improved(x, b2, c2))/np.sqrt(sy**2 + sx**2 * dy_dx**2), '.', color='royalblue') 
axs[1, 1].set_xlabel("Area [$km²$]", fontsize=15)
axs[1, 1].set_ylabel("Residuals", fontsize=15)
axs[1, 1].axhline(0, color='black', lw=1.5, ls='--')
axs[0, 1].legend(fontsize=15, frameon=False)
plt.savefig("/home/ali/Figures/fig12.png", dpi=500, bbox_inches='tight')
plt.show()
area_half_V = AV_power_fit_improved(0.5, b2, c2)
error_area_half_V = np.sqrt((0.5**b2 * c2_err)**2 + (c2 * 0.5**b2 * np.log(0.5) * b2_err)**2)
print(f"\n Volume of glacier with area A=0.5 km²: {area_half_V:.5f} +/- {error_area_half_V:.5f}")
