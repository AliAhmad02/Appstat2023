import pandas as pd
import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2
import matplotlib.pyplot as plt
from scipy.special import erf

dataframe = pd.read_csv('AppStat2023/Exam 2024/data_LargestPopulation.csv', header=0)
Year, PopIndia, PopChina = dataframe.values.T

mask = (Year >= 1963) & (Year <= 1973)

def lin_fit(x, a, b):
    return a * x + b

lin_india_err = 1_300_000
lstsq_lin_india = LeastSquares(Year[mask], PopIndia[mask], lin_india_err, lin_fit)
minuit_lin_india = Minuit(lstsq_lin_india, a=10**7, b=10**7)
minuit_lin_india.migrad()
minuit_lin_india.hesse()
a1, b1 = minuit_lin_india.values[:]
a1_err, b1_err = minuit_lin_india.errors[:]
chi2_lin_india = minuit_lin_india.fval
n_dof_lin_india = len(Year[mask]) - minuit_lin_india.nfit 
p_val_lin_india = chi2.sf(chi2_lin_india, n_dof_lin_india)

lin_india_linspace = np.linspace(1963, 1973, 10_000)
lin_india_fit_vals = lin_fit(lin_india_linspace, a1, b1)

fig, ax = plt.subplots(1, 1, figsize=(6,4))
ax.errorbar(Year[mask], PopIndia[mask], yerr=lin_india_err, fmt='.k', label='Data')
ax.plot(lin_india_linspace, lin_india_fit_vals, label='Fit')
ax.text(0.05, 0.95, f"$a≈${a1:.2e}$\pm${a1_err:.2e}\n$b≈${b1:.2e}$\pm${b1_err:.2e}\n$\chi²$≈{chi2_lin_india:.2f}\n$n_{{DOF}}$={n_dof_lin_india}\n$p(\chi², n_{{DOF}})$≈{p_val_lin_india:.2f}", transform=ax.transAxes, ma='left', va='top', ha='left', fontsize=13, family='monospace')
ax.set_xlabel("Year", fontsize=15)
ax.set_ylabel("Population", fontsize=15)
plt.legend(fontsize=13, frameon=False, loc='lower right')
# plt.savefig("/home/ali/Figures/fig5.png", bbox_inches='tight', dpi=500)
plt.show()
print(f"\n Goodness of linear fit, India: p(chi2={chi2_lin_india:.3f}, ndof={n_dof_lin_india})={p_val_lin_india:.3f}")

pop_error = 1_000_000
mask1 = Year >= 1995

def polyonmial3_fit(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d 

lstsq_india_full = LeastSquares(Year[mask1], PopIndia[mask1], pop_error, polyonmial3_fit)
minuit_india_full = Minuit(lstsq_india_full, a=10**3, b=10**3, c=10**3, d=10**3)
minuit_india_full.migrad()
minuit_india_full.hesse()
a2, b2, c2, d2 = minuit_india_full.values[:]
a2_err, b2_err, c2_err, d2_err = minuit_india_full.errors[:]
chi2_india_full = minuit_india_full.fval
n_dof_india_full = len(Year[mask1]) - minuit_india_full.nfit 
p_val_india_full = chi2.sf(chi2_india_full, n_dof_india_full)

india_linspace_full = np.linspace(1995, 2021, 10_000)
india_fit_vals_full = polyonmial3_fit(india_linspace_full, a2, b2, c2, d2)
fig, ax = plt.subplots(1, 1, figsize=(8,5))
ax.errorbar(Year[mask1], PopIndia[mask1], yerr=pop_error, fmt='.k', label='Data')
ax.plot(india_linspace_full, india_fit_vals_full, label='Fit')
ax.text(0.05, 0.95, f"$a≈${a2:.2e}$\pm${a2_err:.2e}\n$b≈${b2:.2e}$\pm${b2_err:.2e}\n$c≈${c2:.2e}$\pm${c2_err:.2e}\n$d≈${d2:.2e}$\pm${d2_err:.2e}\n$\chi²$≈{chi2_india_full:.2f}\n$n_{{DOF}}$={n_dof_india_full}\n$p(\chi², n_{{DOF}})$≈{p_val_india_full:.2f}", transform=ax.transAxes, ma='left', va='top', ha='left', fontsize=13, family='monospace')
ax.set_xlabel("Year", fontsize=15)
ax.set_ylabel("Population", fontsize=15)
plt.legend(fontsize=13, frameon=False, loc='lower right')
plt.savefig("/home/ali/Figures/fig16.png", bbox_inches='tight', dpi=500)
plt.show()

start_year = 2018
mask1 = mask1 = Year >= start_year
lstsq_china_full = LeastSquares(Year[mask1], PopChina[mask1], pop_error, lin_fit)
minuit_china_full = Minuit(lstsq_china_full, a=10**5, b=10)
minuit_china_full.migrad()
minuit_china_full.hesse()
a3, b3 = minuit_china_full.values[:]
a3_err, b3_err = minuit_china_full.errors[:]
chi2_china_full = minuit_china_full.fval
n_dof_china_full = len(Year[mask1]) - minuit_china_full.nfit 
p_val_china_full = chi2.sf(chi2_china_full, n_dof_china_full)

china_linspace_full = np.linspace(start_year, 2021, 10_000)
china_fit_vals_full = lin_fit(china_linspace_full, a3, b3)
fig, ax = plt.subplots(1, 1, figsize=(8,5))
ax.errorbar(Year[mask1], PopChina[mask1], yerr=pop_error, fmt='.k', label='Data')
ax.plot(china_linspace_full, china_fit_vals_full, label='Fit')
ax.text(0.05, 0.95, f"$a≈${a3:.2e}$\pm${a3_err:.2e}\n$b≈${b3:.2e}$\pm${b3_err:.2e}\n$\chi²$≈{chi2_china_full:.2f}\n$n_{{DOF}}$={n_dof_china_full}\n$p(\chi², n_{{DOF}})$≈{p_val_china_full:.2f}", transform=ax.transAxes, ma='left', va='top', ha='left', fontsize=13, family='monospace')
ax.set_xlabel("Year", fontsize=15)
ax.set_ylabel("Population", fontsize=15)
plt.legend(fontsize=13, frameon=False, loc='lower right')
# plt.savefig("/home/ali/Figures/fig17.png", bbox_inches="tight", dpi=500)
plt.show()

new_linspace = np.linspace(start_year, 2050, 100_00)
poly_vals_future = polyonmial3_fit(a2, b2, c2, d2, new_linspace)
lin_vals_future = lin_fit(a3, b3, new_linspace)
year_takeover = new_linspace[poly_vals_future>=lin_vals_future][0]
