import numpy as np
import pandas as pd
from scipy.stats import normaltest, anderson, pearsonr, spearmanr, chi2
import matplotlib.pyplot as plt
from iminuit import Minuit

dataframe = pd.read_csv('AppStat2023/DataAndCodeForProblemSet/data_CountryScores.csv', header=0, index_col=None)
Country, GDP, PopSize, HappinessI, EconomicFreedomI, PressFreedomI, EducationI = dataframe.values.T

GDP = dataframe["GDP"].values
GDP_mean = dataframe["GDP"].mean()
GDP_std = dataframe["GDP"].std()
GDP_q1 = dataframe["GDP"].quantile(0.1587)
GDP_q2 = dataframe["GDP"].quantile(0.8413)
print(f"\n GDP mean {GDP_mean} \n GDP standard deviation {GDP_std} \n GDP 15.87% quantile {GDP_q1} \n GDP 84.13% quantile {GDP_q2}")

log10_popsize = np.log10(dataframe["PopSize"].values)
print("\n Yes, it is from a Gaussian, see the two tests")
print(normaltest(log10_popsize))
print(anderson(log10_popsize))

happiness = dataframe["Happiness-index"].values
education = dataframe["Education-index"].values

pearson_test = pearsonr(happiness, education)
spearman_test = spearmanr(happiness, education)
print(f"\n Pearson correlation and test {pearson_test}")
print(f"\n Spearman correlation and test {spearman_test}")

x = GDP
y = happiness



def log_func(x, a, b):
    return  a * np.log(x) + b

sy = 600

def chi2_log(a, b):
    y_fit = log_func(x, a, b)
    chi2 = chi2 = np.sum(((y - y_fit)**2/sy**2))
    return chi2

minuit_log = Minuit(chi2_log, a=600, b=50)
minuit_log.errordef = 1.0
minuit_log.migrad()
a, b = minuit_log.values[:]
print(a, b)
a_error, b_error = minuit_log.errors[:]
chi2_val = minuit_log.fval
ndof = len(GDP) - minuit_log.nfit
p_val_chi2 = chi2.sf(chi2_val, ndof)
print(f"\n chi2 p-value {p_val_chi2}")
x_lin = np.linspace(20, 135_000, 130_000)

fig, ax = plt.subplots(1, 1, figsize=(8,5))
ax.errorbar(GDP, happiness, yerr=sy, fmt='.k', label='Data')
ax.plot(x_lin, log_func(x_lin, a, b), lw=2, label='Fit')
ax.set_xlabel("GDP", fontsize=15)
ax.set_ylabel("Happiness Index", fontsize=15)
ax.text(0.95, 0.6, f"$a$≈{a:.2f}$\pm${a_error:.2f}\n$b$≈{b:.2f}$\pm${b_error:.2f}\n$\chi²$≈{chi2_val:.2f}\n$p(\chi²)$≈{p_val_chi2:.2f}", transform=ax.transAxes, ma='left', va='top', ha='right', fontsize=15, family='monospace')
plt.legend(fontsize=15, frameon=False, loc='lower right')
plt.show()
print(f"\n Estimated uncertainty on happiness index is 600 (fit p-value)")
