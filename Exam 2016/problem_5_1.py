import numpy as np
from scipy.stats import chi2
from iminuit import Minuit
from iminuit.cost import LeastSquares
import matplotlib.pyplot as plt


data = np.genfromtxt(r"AppStat2023/Exam 2016/problem 5.1 data file.txt",  delimiter=",")
month, _, income, income_err = data.T
first_year_mask = month <= 12
month_first_year = month[first_year_mask]
income_first_year = income[first_year_mask]
income_err_first_year = income_err[first_year_mask]
first_year_income_avg = np.average(income_first_year, weights=1/income_err_first_year**2)
chi2_first_year = np.sum(((income_first_year - first_year_income_avg) / income_err_first_year)**2)
n_dof_first_year = len(income_first_year) - 1
p_val_first_year = chi2.sf(chi2_first_year, n_dof_first_year)
print(f"\n p-value first year constant fit {p_val_first_year:.3f}")

def lin_fit(x, a, b):
    return a * x + b


lstsq_lin_year = LeastSquares(month_first_year, income_first_year, income_err_first_year, lin_fit)
minuit_lin_year = Minuit(lstsq_lin_year, a=1, b=1)
minuit_lin_year.migrad()
minuit_lin_year.hesse()
a, b = minuit_lin_year.values[:]
a_err, b_err = minuit_lin_year.errors[:]
year_lin = np.linspace(1, 12, 10_000)
chi2_year_lin = minuit_lin_year.fval
n_dof_year_lin = len(income_first_year) - minuit_lin_year.nfit
p_val_year_lin = chi2.sf(chi2_year_lin, n_dof_year_lin)
print(f"\n p-value first year linear fit {p_val_year_lin:.3f}")

plt.figure(figsize=(6,5))
plt.errorbar(month_first_year, income_first_year, income_err_first_year, fmt='.k', capsize=2)
plt.plot(year_lin, lin_fit(year_lin, a, b))
plt.show()

p_vals = []
months_vals = []
p_val = 1
i = 13
p_val_cutoff = 0.01
while p_val > p_val_cutoff:
    lstsq = LeastSquares(month[:i], income[:i], income_err[:i], lin_fit)
    minuit_lin = Minuit(lstsq, a=1, b=1)
    minuit_lin.migrad()
    minuit_lin.hesse()
    chi2_lin = minuit_lin.fval
    n_dof_lin = len(month[:i]) - minuit_lin.nfit
    p_val = chi2.sf(chi2_lin, n_dof_lin)
    p_vals.append(p_val)
    months_vals.append(month[i-1])
    i += 1
plt.figure(figsize=(6,5))
plt.plot(months_vals, p_vals, '.')
plt.axhline(0.01, ls='--', lw=2, color='k')
plt.xlabel("Month cutoff")
plt.ylabel("p-value")
plt.show()
print(p_vals)