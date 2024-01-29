import numpy as np
from scipy.stats import chi2, t

speed = np.array([1532, 1458, 1499, 1394, 1432, 1565, 1474, 1440, 1507])
speed_err = np.array([67, 55, 74, 129, 84, 19, 10, 17, 14])

speed_wmean = np.average(speed, weights=1/speed_err**2)
speed_wmean_err = np.sqrt(1 / np.sum(1/speed_err**2))

speed_wmean_ffive = np.average(speed[:5], weights=1/speed_err[:5]**2)
speed_wmean_lfour = np.average(speed[5:], weights=1/speed_err[5:]**2)

speed_wmean_err_ffive = np.sqrt(1 / np.sum(1/speed_err[:5]**2))
speed_wmean_err_lfour = np.sqrt(1 / np.sum(1/speed_err[5:]**2))

print(f"\n Combined result for the speed: {speed_wmean:.3f} +/- {speed_wmean_err:.3f}")
print(f"\n Combined result first five points: {speed_wmean_ffive:.3f} +/- {speed_wmean_err_ffive:.3f}")
print(f"\n Combined result for the last four points: {speed_wmean_lfour:.3f} +/- {speed_wmean_err_lfour:.3f}")

chi2_speed = np.sum(((speed - speed_wmean) / (speed_err))**2)

chi2_speed_ffive = np.sum(((speed[:5] - speed_wmean_ffive) / (speed_err[:5]))**2)
chi2_speed_lfour = np.sum(((speed[5:] - speed_wmean_lfour) / (speed_err[5:]))**2)

ndof_speed = len(speed) - 1

ndof_speed_ffive = len(speed[:5]) - 1
ndof_speed_lfour = len(speed[5:]) - 1

p_val_speed = chi2.sf(chi2_speed, ndof_speed)
p_val_speed_ffive = chi2.sf(chi2_speed_ffive, ndof_speed_ffive)
p_val_speed_lfour = chi2.sf(chi2_speed_lfour, ndof_speed_lfour)

speed_mean = np.mean(speed)
speed_mean_err = np.std(speed, ddof=1) / np.sqrt(len(speed))

exact_val = 1481
t_val = (exact_val - speed_mean) / speed_mean_err
df_t_test = len(speed) - 1
p_val_t = t.sf(t_val, df_t_test)

print(f"\n Consistency check speed: p(chi2={chi2_speed:.3f}, ndof={ndof_speed}): {p_val_speed:.4f}")
print(f"\n Consistency check speed first 5 points: p(chi2={chi2_speed_ffive:.3f}, ndof={ndof_speed_ffive}): {p_val_speed_ffive:.3f}")
print(f"\n Consistency check speed last 4 points: p(chi2={chi2_speed_lfour:.3f}, ndof={ndof_speed_lfour}): {p_val_speed_lfour}")

print(f"\n Final estimate of the speed: {speed_mean:.3f} +/- {speed_mean_err:.3f}")
print(f"\n Check estimate against true value: p(t={t_val:.3f}, df={df_t_test}): {p_val_t:.3f}")
