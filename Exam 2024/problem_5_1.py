import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

data = np.genfromtxt('AppStat2023/Exam 2024/data_SignalDetection.csv', delimiter = ',', skip_header=1)
index = data[:,0].astype(int)
P     = data[:,1]
R     = data[:,2]
freq  = data[:,3]
type  = data[:,4].astype(int)

n_bins = 340
xmin = 0
xmax = 16
# n_bins = 120
# xmin = 1.2
# xmax = 1.6

freq_control = freq[:-20_000]

# mask1 = (freq_control >= 1.2) & (freq_control <= 1.6)
# freq_control = freq_control[mask1]
bin_width = (xmax - xmin) / n_bins
counts, bin_edges = np.histogram(freq_control, bins=n_bins, range=(xmin, xmax))
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
mask = counts > 0
def gauss_pdf(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-0.5 * (x-mu)**2/sigma**2)

def exp_pdf(x, tau):
    return 1 / tau * np.exp(- x / tau)

def fit_func(x, Ngauss, Nexp, mu, sigma, tau):
    return Ngauss * bin_width * gauss_pdf(x, mu, sigma) + Nexp * bin_width * exp_pdf(x, tau)

lstsq_H = LeastSquares(bin_centers[mask], counts[mask], np.sqrt(counts[mask]), fit_func)
minuit_H = Minuit(lstsq_H, Ngauss=10**4, Nexp=10**5, mu=1.42, sigma=0.1, tau=1)
minuit_H.migrad()
minuit_H.hesse()

Ngauss, Nexp, mu, sigma, tau = minuit_H.values[:]
Ngauss_err, Nexp_err, mu_err, sigma_err, tau_err = minuit_H.errors[:]
chi2_H = minuit_H.fval
ndof_H = len(counts[mask]) - minuit_H.nfit
p_val_H = chi2.sf(chi2_H, ndof_H)
linspace_H = np.linspace(xmin, xmax, 100_000)
fvals_H = fit_func(linspace_H, Ngauss, Nexp, mu, sigma, tau)
fig, ax = plt.subplots(1, 1, figsize=(12,7))
ax.hist(freq_control, bins=n_bins, range=(xmin, xmax), histtype='step', label='Histogram', lw=1.3)
# ax.text(0.05, 0.99, f"$N_{{exp}}≈${Nexp:.2e}$\pm${Nexp_err:.2e}\n$N_{{gauss}}≈${Ngauss:.2e}$\pm${Ngauss_err:.2e}\n$\mu≈${mu:.3f}$\pm${mu_err:.3f}\n$\sigma≈${sigma:.3f}$\pm${sigma_err:.3f}\n$𝜏$≈{tau:.3f}$\pm${tau_err:.3f}\n$\chi²$≈{chi2_H:.3f}\n$n_{{DOF}}$={ndof_H}\n$p(\chi², n_{{DOF}})$≈{p_val_H:.3f}", transform=ax.transAxes, ma='left', va='top', ha='left', fontsize=13, family='monospace')
# ax.plot(linspace_H, fvals_H, label='Fit', lw=1.9)
ax.set_xlabel("Frequency [GHz]", fontsize=15)
plt.legend(fontsize=15, frameon=False)
# plt.savefig("/home/ali/Figures/fig7.png", bbox_inches="tight", dpi=500)
# plt.savefig("/home/ali/Figures/fig6.png", bbox_inches="tight", dpi=500)
plt.show()

def calc_separation(x, y):  
    d = np.abs(np.mean(x)-np.mean(y))/np.sqrt(np.std(x, ddof=1)**2 + np.std(y, ddof=1)**2)
    return d

def fisher_signal_noise(noise_var1, signal_var1, noise_var2, signal_var2):
    mu_n = np.array([noise_var1.mean(), noise_var2.mean()])
    mu_s = np.array([signal_var1.mean(), signal_var2.mean()])

    mat_n = np.stack((noise_var1, noise_var2))
    mat_s = np.stack((signal_var1, signal_var2))
    
    cov_n = np.cov(mat_n, ddof=1)
    cov_s = np.cov(mat_s, ddof=1)
    cov_sum_inv = np.linalg.inv(cov_n + cov_s)
    
    wf = cov_sum_inv @ (mu_s-mu_n)
    
    fisher_n = np.dot(wf, mat_n)
    fisher_s = np.dot(wf, mat_s)
    return fisher_n, fisher_s, wf

P_s = P[type==1]
P_n = P[type==0]

R_s = R[type==1]
R_n = R[type==0]

sep_P = calc_separation(P_s, P_n)
sep_R = calc_separation(R_s, R_n)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.hist(P_n, bins=350, histtype='stepfilled', color='blue', alpha=0.7)
ax1.hist(P_s, bins=45, histtype='stepfilled', color='red', alpha=0.7)
ax1.set_xlabel("Phase", fontsize=15)
ax1.text(0.05, 0.9, fr'$\Delta_P=${sep_P:.2f}', horizontalalignment='left',
     verticalalignment='center', transform=ax1.transAxes, fontsize=15)

ax2.hist(R_n, bins=350, histtype='stepfilled', color='blue', alpha=0.7, label="Noise")
ax2.hist(R_s, bins=45, histtype='stepfilled', color='red', alpha=0.7, label="Signal")
ax2.set_xlabel("Resonance", fontsize=15)
ax2.text(0.95, 0.9, fr'$\Delta_R=${sep_R:.2f}', horizontalalignment='right',
     verticalalignment='center', transform=ax2.transAxes, fontsize=15)
ax2.legend(fontsize=15, frameon=False, loc='upper left')
# plt.savefig("/home/ali/Figures/fig8.png", bbox_inches="tight", dpi=500)
plt.show()

fisher_n, fisher_s, wf = fisher_signal_noise(P_n, P_s, R_n, R_s)
w0 = 6
fisher_n += w0
fisher_s += w0
fisher_sep = calc_separation(fisher_n, fisher_s)
fig, ax = plt.subplots(1, 1, figsize=(8,5))
ax.hist(fisher_n, bins=350, histtype='step', color='blue', label='Noise')
ax.hist(fisher_s, bins=45, histtype='step', color='red', label='Signal')
ax.text(0.05, 0.9, fr'$\Delta_{{PR}}=${fisher_sep:.2f}', horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes, fontsize=15)
ax.set_xlabel("Fisher Discriminant", fontsize=15)
plt.legend(fontsize=15, frameon=False)
# plt.savefig("/home/ali/Figures/fig12.png", bbox_inches="tight", dpi=500)
plt.show()

a1 = np.append(fisher_n, fisher_s)
b1 = np.append(np.zeros(len(fisher_n)), np.ones(len(fisher_s)))

a2 = np.append(P_n, P_s)
b2 = np.append(np.ones(len(P_n)), np.zeros(len(P_s)))

a3 = np.append(R_n, R_s)
b3 = np.append(np.zeros(len(R_n)), np.ones(len(R_s)))

FPR_fisher, TPR_fisher, thresholds_fisher = roc_curve(b1, a1)
FPR_P, TPR_P, thresholds_P = roc_curve(b2, a2)
FPR_R, TPR_R, thresholds_R = roc_curve(b3, a3)
AUC_fisher = roc_auc_score(b1, a1)
AUC_P = roc_auc_score(b2, a2)
AUC_R = roc_auc_score(b3, a3)

print(f"\n Weights for Fisher discrimiant: {wf}")

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(FPR_fisher, TPR_fisher, label=f"Fisher (AUC={AUC_fisher:.2f})")
ax.plot(FPR_P, TPR_P, label=f"Phase (AUC={AUC_P:.2f})")
ax.plot(FPR_R, TPR_R, label=f"Resonance (AUC={AUC_R:.2f})")
ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls='dashed', label="Random classifier", color='black')
ax.set_xlabel("False positive rate", fontsize=15)
ax.set_ylabel("True positive rate", fontsize=15)
plt.legend(fontsize=15, frameon=False)
# plt.savefig("/home/ali/Figures/fig13.png", bbox_inches="tight", dpi=500)
plt.show()

TPR_choice = 0.85
threshold_cutoff = thresholds_fisher[TPR_fisher>=0.85][0]
print(f"\n Fisher discriminant cutoff: {threshold_cutoff:.3f}")

fisher_control_full = P[:-20_000] * wf[0] + R[:-20_000] * wf[1] + w0
freq_control_signal_fisher = freq_control[fisher_control_full>=threshold_cutoff]
n_bins1 = 530
xmin1 = 0
xmax1 = 14
bin_width1 = (xmax1 - xmin1) / n_bins1
counts1, bin_edges1 = np.histogram(freq_control_signal_fisher, bins=n_bins1, range=(xmin1, xmax1))
bin_centers1 = (bin_edges1[1:] + bin_edges1[:-1]) / 2
mask1 = counts1 > 0

def fit_func(x, Ngauss, Nexp, mu, sigma, tau):
    return Ngauss * bin_width1 * gauss_pdf(x, mu, sigma) + Nexp * bin_width1 * exp_pdf(x, tau)

lstsq_fisher = LeastSquares(bin_centers1[mask1], counts1[mask1], np.sqrt(counts1[mask1]), fit_func)
minuit_fisher = Minuit(lstsq_fisher, Ngauss=10**4, Nexp=10**5, mu=1.42, sigma=0.1, tau=1)
minuit_fisher.migrad()
minuit_fisher.hesse()

Ngauss1, Nexp1, mu1, sigma1, tau1 = minuit_fisher.values[:]
Ngauss1_err, Nexp1_err, mu1_err, sigma1_err, tau1_err = minuit_fisher.errors[:]
chi2_fisher = minuit_fisher.fval
ndof_fisher = len(counts1[mask1]) - minuit_fisher.nfit
p_val_fisher = chi2.sf(chi2_fisher, ndof_fisher)
linspace_fisher = np.linspace(xmin1, xmax1, 100_000)
fvals_fisher = fit_func(linspace_fisher, Ngauss1, Nexp1, mu1, sigma1, tau1)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.hist(freq_control_signal_fisher, bins=n_bins1, histtype='step', label='Histogram', lw=1.3)
ax.text(0.99, 0.05, f"$N_{{exp}}≈${Nexp1:.2e}$\pm${Nexp1_err:.2e}\n$N_{{gauss}}≈${Ngauss1:.2e}$\pm${Ngauss1_err:.2e}\n$\mu≈${mu1:.3f}$\pm${mu1_err:.3f}\n$\sigma≈${sigma1:.3f}$\pm${sigma1_err:.3f}\n$𝜏$≈{tau1:.3f}$\pm${tau1_err:.3f}\n$\chi²$≈{chi2_fisher:.3f}\n$n_{{DOF}}$={ndof_fisher}\n$p(\chi², n_{{DOF}})$≈{p_val_fisher:.3f}", transform=ax.transAxes, ma='left', va='bottom', ha='right', fontsize=13, family='monospace')
ax.plot(linspace_fisher, fvals_fisher, ls='dashed', label='Fit')
ax.set_xlabel("Frequency [GHz]", fontsize=15)
plt.legend(fontsize=15, frameon=False)
# plt.savefig("/home/ali/Figures/fig14.png", bbox_inches='tight', dpi=500)
plt.show()

freq_real = freq[-20_000:]
fisher_real_full = P[-20_000:] * wf[0] + R[-20_000:] * wf[1] + w0
freq_real_signal_fisher = freq_real[fisher_real_full>=threshold_cutoff]

xmin2, xmax2 = 0.1, 1
n_bins2 = 200

bin_width2 = (xmax2 - xmin2) / n_bins2
counts2, bin_edges2 = np.histogram(freq_real, bins=n_bins2, range=(xmin2, xmax2))
bin_centers2 = (bin_edges2[1:] + bin_edges2[:-1]) / 2
mask2 = counts2 > 0

# fig, ax = plt.subplots(1, 1, figsize=(8, 5))
# ax.hist(freq_real, bins=n_bins2, histtype='step', label='Histogram', lw=1.3)
# ax.set_xlabel("Frequency [GHz]", fontsize=15)
# plt.legend(fontsize=15, frameon=False)
# plt.show()

xmin3, xmax3 = 0.1, 1
n_bins3 = 50

bin_width3 = (xmax3 - xmin3) / n_bins3
counts3, bin_edges3 = np.histogram(freq_real_signal_fisher, bins=n_bins3, range=(xmin3, xmax3))
bin_centers3 = (bin_edges3[1:] + bin_edges3[:-1]) / 2
mask3 = counts3 > 0

def fit_func(x, Ngauss, Nexp, mu, sigma, tau):
    return Ngauss * bin_width3 * gauss_pdf(x, mu, sigma) + Nexp * bin_width3 * exp_pdf(x, tau)

lstsq_fisher_real = LeastSquares(bin_centers3[mask3], counts3[mask3], np.sqrt(counts3[mask3]), fit_func)
minuit_fisher_real = Minuit(lstsq_fisher_real, Ngauss=100, Nexp=100, mu=0.1, sigma=0.1, tau=1)
minuit_fisher_real.migrad()
minuit_fisher_real.hesse()

Ngauss3, Nexp3, mu3, sigma3, tau3 = minuit_fisher_real.values[:]
Ngauss3_err, Nexp3_err, mu3_err, sigma3_err, tau3_err = minuit_fisher_real.errors[:]
chi2_fisher_real = minuit_fisher_real.fval
ndof_fisher_real = len(counts3[mask3]) - minuit_fisher_real.nfit
p_val_fisher_real = chi2.sf(chi2_fisher_real, ndof_fisher_real)
linspace_fisher_real = np.linspace(xmin3, xmax3, 10_000)
fvals_fisher_real = fit_func(linspace_fisher_real, Ngauss3, Nexp3, mu3, sigma3, tau3)

fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(14, 5))
ax1.hist(freq_real_signal_fisher, bins=n_bins3, range=(xmin3, xmax3), histtype='step', label='Histogram', lw=1.3)
ax1.text(0.99, 0.99, f"$N_{{exp}}≈${Nexp3:.2e}$\pm${Nexp3_err:.2e}\n$N_{{gauss}}≈${Ngauss3:.2e}$\pm${Ngauss3_err:.2e}\n$\mu≈${mu3:.3f}$\pm${mu3_err:.3f}\n$\sigma≈${sigma3:.3f}$\pm${sigma3_err:.3f}\n$𝜏$≈{tau3:.3f}$\pm${tau3_err:.3f}\n$\chi²$≈{chi2_fisher_real:.3f}\n$n_{{DOF}}$={ndof_fisher_real}\n$p(\chi², n_{{DOF}})$≈{p_val_fisher_real:.3f}", transform=ax1.transAxes, ma='left', va='top', ha='right', fontsize=13, family='monospace')
ax1.plot(linspace_fisher_real, fvals_fisher_real, ls='dashed', label='Fit')
ax1.set_xlabel("Frequency [GHz]", fontsize=15)
ax2.hist(freq_real, bins=n_bins2, histtype='step', label='Histogram', lw=1.3)
ax2.set_xlabel("Frequency [GHz]", fontsize=15)
ax1.legend(fontsize=15, loc='lower left')
# plt.savefig("/home/ali/Figures/fig15.png", bbox_inches='tight', dpi=500)
plt.show()

pred = np.array([1 if a_val>=threshold_cutoff else 0 for a_val in a1])
confusion = confusion_matrix(b1, pred)
FNR = confusion[1, 0] / (confusion[1, 0] + confusion[1, 1])
print(f"\n False negative rate: {FNR:.2f}")

n_rejected_signal = len(freq_real[(fisher_real_full<threshold_cutoff) & (freq_real >=xmin3) & (freq_real <= xmax3)])
total_entries = Ngauss3 * TPR_choice + FNR * n_rejected_signal
total_entries_err = np.sqrt(TPR_choice**2 * Ngauss3_err**2 + FNR**2 * np.sqrt(n_rejected_signal))
print(f"\n Estimated signal entries taking errors into account: {total_entries:.3f} +/- {total_entries_err:.3f}")
