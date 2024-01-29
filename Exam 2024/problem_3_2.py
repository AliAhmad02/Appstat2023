import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from iminuit import Minuit
from iminuit.cost import LeastSquares

C_pdf = 1 / (3*np.pi)

def PDF(x):
    return C_pdf * (np.arctan(x) + np.pi/2)

xmin = -3
xmax = 3
ymin = PDF(xmin)
ymax = PDF(xmax)

x_vals = np.random.uniform(xmin, xmax, 1000)
y_vals = np.random.uniform(ymin, ymax, 1000)
f_vals = PDF(x_vals)
accepted_vals = x_vals[y_vals < f_vals]
N = 100
accepted_vals = np.random.choice(accepted_vals, size=100)
n_bins = 25
counts, bin_edges = np.histogram(accepted_vals, bins=n_bins, range=(xmin, xmax))
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
binwidth = (xmax - xmin) / n_bins
plt.figure(figsize=(6,4))
plt.hist(accepted_vals, bins=n_bins, range=(xmin, xmax), histtype='step', lw=1.5)
plt.xlabel("Random numbers according to PDF", fontsize=13)
plt.show()

mask = counts > 0
def PDF_fit(x, C):
    return N * C * binwidth * (np.arctan(x) + np.pi/2)

lstsq = LeastSquares(bin_centers[mask], counts[mask], np.sqrt(counts[mask]), PDF_fit)
minuit_lstsq = Minuit(lstsq, C=1/(3*np.pi))
minuit_lstsq.migrad()
minuit_lstsq.hesse()
C1, = minuit_lstsq.values[:]
C1_err, = minuit_lstsq.errors[:]
chi2_lstsq = minuit_lstsq.fval
ndof_lstsq = len(bin_centers[mask]) - minuit_lstsq.nfit
p_val_lstsq = chi2.sf(chi2_lstsq, ndof_lstsq)
print(f"\n C fit parameter value: {C1:.3f} +/- {C1_err:.3f}")
print(f"\n Fit goodness p(chi2={chi2_lstsq:.3f}, ndof={ndof_lstsq:.3f})")

lstsq_linspace = np.linspace(xmin, xmax, 1000)
lstsq_fit_vals = PDF_fit(lstsq_linspace, C1)

fig, ax = plt.subplots(1, 1, figsize=(8,5))

ax.hist(accepted_vals, histtype='step', bins=n_bins, range=(xmin, xmax))
ax.plot(lstsq_linspace, lstsq_fit_vals)
ax.set_xlabel(r"Numbers according to PDF", fontsize=15)
ax.text(0.05, 0.95, f"$C≈${C1:.3f}$\pm${C1_err:.3f}\n$\chi²$≈{chi2_lstsq:.3f}\n$p(\chi²)$≈{p_val_lstsq:.3f}", transform=ax.transAxes, ma='left', va='top', ha='left', fontsize=13, family='monospace')
# plt.savefig("/home/ali/Figures/fig4.png", bbox_inches='tight', dpi=500)
plt.show()
