import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import normaltest, pearsonr
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

# import re

# with open(r"AppStat2023/Exam 2016/problem 4.1 data file.txt") as f:
#     content = f.read()

# comma_separated = re.sub(r"[ \t]{3,}", ",", content)
# with open(r"AppStat2023/Exam 2016/problem 4.1 data file.txt", "w") as f:
#     f.write(comma_separated)

def calc_separation(x, y):  
    d = np.abs(np.mean(x)-np.mean(y))/np.sqrt(np.std(x, ddof=1)**2 + np.std(y, ddof=1)**2)
    return d

data = pd.read_csv(r"AppStat2023/Exam 2016/problem 4.1 data file.txt", delimiter=",")
Status, PatientID, A, B, C = data.values.T
A_healthy = A[Status==0]
A_ill = A[Status==1]
B_healthy = B[Status==0]
B_ill = B[Status==1]
C_healthy = C[Status==0]
C_ill = C[Status==1]
p_val_A_healthy_norm = normaltest(A_ill).pvalue
print(f"\n A healthy is normal distributed. See p-value {p_val_A_healthy_norm:.3f}")
pearson_BC_ill = pearsonr(B_ill, C_ill).statistic
print(f"\n Pearson correlation for BC-ill {pearson_BC_ill:.3f}")

sep_A = calc_separation(A_healthy, A_ill)
sep_B = calc_separation(B_healthy, B_ill)
sep_C = calc_separation(C_healthy, C_ill)

fig = plt.figure(figsize=(10, 10))
gs = fig.add_gridspec(2,2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

ax1.hist(A_healthy, bins=50, histtype='stepfilled', color='blue', alpha=0.7)
ax1.hist(A_ill, bins=50, histtype='stepfilled', color='red', alpha=0.7)
ax1.set_xlabel("A", fontsize=15)
ax1.text(0.05, 0.9, fr'$\Delta_A=${sep_A:.2f}', horizontalalignment='left',
     verticalalignment='center', transform=ax1.transAxes, fontsize=15)


ax2.hist(B_healthy, bins=50, histtype='stepfilled', color='blue', alpha=0.7)
ax2.hist(B_ill, bins=50, histtype='stepfilled', color='red', alpha=0.7)
ax2.set_xlabel("B", fontsize=15)
ax2.text(0.95, 0.9, fr'$\Delta_B=${sep_B:.2f}', horizontalalignment='right',
     verticalalignment='center', transform=ax2.transAxes, fontsize=15)

ax3.hist(C_healthy, bins=50, histtype='stepfilled', color='blue', label='Healthy', alpha=0.7, zorder=0)
ax3.hist(C_ill, bins=50, histtype='stepfilled', color='red', label='Sick', alpha=0.7)
ax3.set_xlabel("C", fontsize=15)
ax3.text(0.05, 0.9, fr'$\Delta_C=${sep_C:.2f}', horizontalalignment='left',
     verticalalignment='center', transform=ax3.transAxes, fontsize=15)
plt.legend(fontsize=15, frameon=False)
plt.show()

def fisher_healthy_sick(healthy_var1, sick_var1, healthy_var2, sick_var2, healthy_var3, sick_var3):
    mu_h = np.array([healthy_var1.mean(), healthy_var2.mean(), healthy_var3.mean()])
    mu_s = np.array([sick_var1.mean(), sick_var2.mean(), sick_var3.mean()])

    mat_h = np.stack((healthy_var1, healthy_var2, healthy_var3))
    mat_s = np.stack((sick_var1, sick_var2, sick_var3))
    
    cov_h = np.cov(mat_h, ddof=1)
    cov_s = np.cov(mat_s, ddof=1)
    cov_sum_inv = np.linalg.inv(cov_h + cov_s)
    
    wf = cov_sum_inv @ (mu_s-mu_h)
    
    fisher_h = np.dot(wf, mat_h)
    fisher_s = np.dot(wf, mat_s)
    return fisher_h, fisher_s, wf

fisher_h, fisher_i, wf = fisher_healthy_sick(
    A_healthy, A_ill, B_healthy, B_ill, C_healthy, C_ill
)
w0 = 30
fisher_h += w0
fisher_i += w0

fisher_sep = calc_separation(fisher_h, fisher_i)
fig, ax = plt.subplots(1, 1, figsize=(8,5))
ax.hist(fisher_h, bins=50, histtype='step', color='blue', label='Healthy')
ax.hist(fisher_i, bins=50, histtype='step', color='red', label='Sick')
ax.text(0.05, 0.9, fr'$\Delta_{{BAT}}=${fisher_sep:.2f}', horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes, fontsize=15)
ax.set_xlabel("Fisher Discriminant", fontsize=15)
plt.legend(fontsize=15, frameon=False)
plt.show()

a = np.append(fisher_h, fisher_i)
b = np.append(np.zeros(len(fisher_h)), np.ones(len(fisher_i)))
FPR, TPR, thresholds_ROC = roc_curve(b, a)
ROC_AUC = roc_auc_score(b, a)
plt.figure(figsize=(8, 5))
plt.plot(FPR, TPR, lw=3.5, color='blue')
plt.xlabel("False positive rate", fontsize=15)
plt.ylabel("True positive rate", fontsize=15)
plt.text(0.5, 0.5, f'AUC={ROC_AUC:.6f}', horizontalalignment='center',verticalalignment='center', fontsize=15)
plt.show()

TPR_99_pct_cutoff = thresholds_ROC[np.where(TPR>=0.99)[0][0]]
pred = np.array([1 if a_val>=TPR_99_pct_cutoff else 0 for a_val in a])
confusion = confusion_matrix(b, pred)
error_rates = confusion / len(Status)
FPR_cut = confusion[0, 1] / (confusion[0, 0] + confusion[0, 1])
FNR_cut = confusion[1, 0] / (confusion[1, 0] + confusion[1, 1])
print(f"\n Type 1 error: {FPR_cut:.3f}\n Type 2 error: {FNR_cut:.3f}")
