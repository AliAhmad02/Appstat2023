from scipy.stats import poisson

mu = 52.8
cutoff_top20_pct = poisson.isf(q=0.2, mu=mu)
frac_cutoff = poisson.sf(k=cutoff_top20_pct, mu=mu)
# We add one when printig because scipy stats gives an
# exclusive cutoff, i.e., >59 as opposed to >=59
print(f"\n Top 20% busiest days cutoff: {cutoff_top20_pct+1}")
print(f"\n Integral of poisson for >=60: {frac_cutoff:.3f}")