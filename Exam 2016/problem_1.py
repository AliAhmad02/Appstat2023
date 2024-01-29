from scipy.stats import binom, poisson, norm
import numpy as np
from scipy.integrate import quad

k_game1 = 1
n_game1 = 4
p_game1 = 1/6
p_game2 = 1/36
n_game2 = 24
k_game2 = 1


p_win_game1 = binom.sf(k=k_game1-1, n=n_game1, p=p_game1)
p_win_game2 = binom.sf(k=k_game2-1, n=n_game2, p=p_game2)
print(f"\n Win probability game 1 {p_win_game1}")
print(f"\n Win probability game 2 {p_win_game2}")
print(f"\n Thus, the first game has better odds.")

print(f"\n IceCube background follows Poisson distribution.")

mu_icecube = 18.9
N_icecube = 1730
k_icecube = 42
p_42_or_more = poisson.sf(k=k_icecube-1, mu=mu_icecube)
p_42_or_more_trials = 1 - (1-p_42_or_more)**N_icecube
print(f"\n Probability of 42 or more events (incl. trial factors) {p_42_or_more_trials}")

heights_mean = 1.68
heights_std = 0.06
height_cutoff = 1.85
frac_taller_women = norm.sf(height_cutoff, loc=heights_mean, scale=heights_std)
print(f"\n Fraction of women taller than 1.85m {frac_taller_women}")
frac_tallest_women = 0.2
cutoff_20_percent = norm.isf(q=frac_tallest_women, loc=heights_mean, scale=heights_std)
normalization_const = 33.2452
renorm_func_mean = lambda x: x * normalization_const * np.exp(-(x-heights_mean)**2/(2*heights_std**2))
mean_top_20_pct = quad(renorm_func_mean, cutoff_20_percent, np.inf)[0]
print(f"\n Average height of top 20% tallest women {mean_top_20_pct:.3f}")
