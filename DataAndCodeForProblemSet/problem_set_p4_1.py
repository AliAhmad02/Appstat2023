import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_curve, roc_auc_score

def calc_separation(x, y):  
    d = np.abs(np.mean(x)-np.mean(y))/np.sqrt(np.std(x, ddof=1)**2 + np.std(y, ddof=1)**2)
    return d

dataframe = pd.read_csv('AppStat2023/DataAndCodeForProblemSet/data_AnorocDisease.csv', header=0)
dataframe = dataframe.rename({'      Status (0: Healthy; 1: Ill; -1: Unknown)': 'Status'}, axis='columns')
dataframe.columns = dataframe.columns.str.strip()
PatientID_h, Temp_h, BloodP_h, Age_h, _ = dataframe[dataframe["Status"]==0].values.T
PatientID_s, Temp_s, BloodP_s, Age_s, _ = dataframe[dataframe["Status"]==1].values.T
PatientID_u, Temp_u, BloodP_u, Age_u, _ = dataframe[dataframe["Status"]==-1].values.T

sep_temp = calc_separation(Temp_h, Temp_s)
sep_bp = calc_separation(BloodP_h, BloodP_s)
sep_age = calc_separation(Age_h, Age_s)

fig = plt.figure(figsize=(10, 10))

gs = fig.add_gridspec(2,2)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

ax1.hist(Temp_h, bins=45, range=(33, 42), histtype='stepfilled', color='blue', alpha=0.7)
ax1.hist(Temp_s, bins=45, range=(33, 42), histtype='stepfilled', color='red', alpha=0.7)
ax1.set_xlim(33, 42)
ax1.set_xlabel("Temperature", fontsize=15)
ax1.text(0.05, 0.9, fr'$\Delta_T=${sep_temp:.2f}', horizontalalignment='left',
     verticalalignment='center', transform=ax1.transAxes, fontsize=15)


ax2.hist(BloodP_h, bins=55, range=(90, 170), histtype='stepfilled', color='blue', alpha=0.7)
ax2.hist(BloodP_s, bins=55, range=(90, 170), histtype='stepfilled', color='red', alpha=0.7)
ax2.set_xlim(90, 170)
ax2.set_xlabel("Blood pressure", fontsize=15)
ax2.text(0.95, 0.9, fr'$\Delta_B=${sep_bp:.2f}', horizontalalignment='right',
     verticalalignment='center', transform=ax2.transAxes, fontsize=15)

ax3.hist(Age_h, bins=70, range=(0, 100), histtype='stepfilled', color='blue', label='Healthy', alpha=0.7, zorder=0)
ax3.hist(Age_s, bins=70, range=(0, 100), histtype='stepfilled', color='red', label='Sick', alpha=0.7)
ax3.set_xlim(0, 100)
ax3.set_xlabel("Age", fontsize=15)
ax3.text(0.05, 0.9, fr'$\Delta_A=${sep_age:.2f}', horizontalalignment='left',
     verticalalignment='center', transform=ax3.transAxes, fontsize=15)
plt.legend(fontsize=15, frameon=False)
plt.show()

ks_hs_age = ks_2samp(Age_h, Age_s)[1]
print(f"\n p-value from KS test on age for sick/healthy {ks_hs_age}")

fig, axs = plt.subplots(1, 3, figsize=(15,5))
axs[0].plot(BloodP_h, Temp_h, '.', color='blue')
axs[0].plot(BloodP_s, Temp_s, '.', color='red')
axs[0].set_xlabel("Blood pressure", fontsize=15)
axs[0].set_ylabel("Temperature", fontsize=15)

axs[1].plot(Age_h, BloodP_h, '.', color='blue')
axs[1].plot(Age_s, BloodP_s, '.', color='red')
axs[1].set_xlabel("Age", fontsize=15)
axs[1].set_ylabel("Blood pressure", fontsize=15)

axs[2].plot(Age_h, Temp_h, '.', color='blue', label='Healthy')
axs[2].plot(Age_s, Temp_s, '.', color='red', label='Sick')
axs[2].set_xlabel("Age", fontsize=15)
axs[2].set_ylabel("Temperature", fontsize=15)
plt.legend(fontsize=15, frameon=False, markerscale=2)
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

fisher_h, fisher_s, wf = fisher_healthy_sick(Temp_h, Temp_s, BloodP_h, BloodP_s, Age_h, Age_s)
fisher_sep = calc_separation(fisher_h, fisher_s)
w0 = -35
fisher_h += w0
fisher_s += w0

fig, ax = plt.subplots(1, 1, figsize=(8,5))
ax.hist(fisher_h, bins=50, histtype='step', color='blue', label='Healthy')
ax.hist(fisher_s, bins=50, histtype='step', color='red', label='Sick')
ax.text(0.05, 0.9, fr'$\Delta_{{BAT}}=${fisher_sep:.2f}', horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes, fontsize=15)
ax.set_xlabel("Fisher Discriminant", fontsize=15)
plt.legend(fontsize=15, frameon=False)
plt.show()

a = np.append(fisher_h, fisher_s)
b = np.append(np.zeros(len(fisher_h)), np.ones(len(fisher_s)))
FPR, TPR, thresholds_ROC = roc_curve(b, a)
ROC_AUC = roc_auc_score(b, a)

plt.figure(figsize=(8, 5))
plt.plot(FPR, TPR, lw=3.5, color='blue')
plt.xlabel("False positive rate", fontsize=15)
plt.ylabel("True positive rate", fontsize=15)
plt.axhline(0.825, xmax=0.18, lw=2, linestyle='dashed', color='black')
plt.axvline(0.1425, ymax=0.79, lw=2, linestyle='dashed', color='black')
plt.text(0.5, 0.5, f'AUC={ROC_AUC:.2f}', horizontalalignment='center',verticalalignment='center', fontsize=15)
plt.show()
fisher_unknown = wf[0] * Temp_u + wf[1] * BloodP_u + wf[2] * Age_u + w0
TPR_80_pct_cutoff = thresholds_ROC[np.where(TPR==0.8025)[0][0]]
age_h_predict = Age_u[fisher_unknown<TPR_80_pct_cutoff]
age_s_predict = Age_u[fisher_unknown>TPR_80_pct_cutoff]
temp_h_predict = Temp_u[fisher_unknown<TPR_80_pct_cutoff]
temp_s_predict = Temp_u[fisher_unknown>TPR_80_pct_cutoff]
bp_h_predict = BloodP_u[fisher_unknown<TPR_80_pct_cutoff]
bp_s_predict = BloodP_u[fisher_unknown>TPR_80_pct_cutoff]
temp_u_3D, age_u_3D = np.meshgrid(Temp_u, Age_u)
disc_plane_bp = -(wf[0] * temp_u_3D + wf[2] * age_u_3D + w0 - TPR_80_pct_cutoff) / wf[1]

fig, ax = plt.subplots(figsize=(12,8), subplot_kw={'projection': '3d'})
surf = ax.plot_surface(temp_u_3D, age_u_3D, disc_plane_bp, shade=False, alpha=0.1, label="FD cutoff plane")
surf._edgecolors2d = surf._edgecolor3d
surf._facecolors2d = surf._facecolor3d
ax.plot(temp_h_predict, age_h_predict, bp_h_predict, '.',label='Healthy', color='blue')
ax.plot(temp_s_predict, age_s_predict, bp_s_predict, '.', label='Sick', color='red')
ax.set_xlabel("Temperature", fontsize=15)
ax.set_ylabel("Age", fontsize=15)
ax.set_zlabel("Blood pressure", fontsize=15)
ax.view_init(elev=2, azim=-74)
plt.legend(fontsize=15, frameon=False, bbox_to_anchor=(1.04, 0.5), loc="center left")
plt.show()

print(f"\n Infections in unknown group estimate {len(age_s_predict)}")
Temp_all = np.append(Temp_h, Temp_s)
p_B_given_A = len(Temp_s[Temp_s==38.5])/len(Temp_s)
p_A = 0.01 
p_B_given_healthy = len(Temp_h[Temp_h==38.5])/len(Temp_h)
p_healthy = 1-p_A
p_B = p_B_given_A * p_A + p_B_given_healthy * p_healthy
p_B_given_A = len(Temp_s[Temp_s==38.5])/len(Temp_s)
p_A_given_B = (p_B_given_A * p_A)/p_B

print(f"\n Probability of being ill given T=38.5 {p_A_given_B}")
