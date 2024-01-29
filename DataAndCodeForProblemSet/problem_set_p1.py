import numpy as np
from scipy.stats import norm, binom

mu = 50
sigma = 20

x_start = 55
x_end = 65

fraction = 1 - norm.cdf(x_start, mu, sigma) - norm.sf(x_end, mu, sigma)

print(f"\n Fraction of students in interval [55, 65] \n {fraction}")

err_on_mean = sigma / np.sqrt(120)
print(f"\n Error on mean \n {err_on_mean}")

p_above_70 = norm.sf(70, mu, sigma)
p_uncorr_above_70 = p_above_70**2
p_corr_above_70 = p_above_70**2 + 0.75 * p_above_70*(1-p_above_70)

print(f"\n Probability of getting above 70 in both tests uncorrelated \n {p_uncorr_above_70}.")
print(f"\n Probability of getting above 70 in both tests correlated \n {p_corr_above_70}.")

k_douzaine = 8
n_douzaine = 20
p_douzaine = 12/37
p_douzaine_total = 0
for i in range(k_douzaine, n_douzaine + 1):
    p_douzaine_total += binom.pmf(k=i, n=n_douzaine, p=p_douzaine)
print(f"\n Probability of getting 8 or more douzaine \n {p_douzaine_total}")
