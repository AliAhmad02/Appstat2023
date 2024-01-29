import numpy as np

print(f"\n The relative uncertainty of L should be twice the relative uncertainty of r")
speed_array = 100 * np.array([3.61, 2.00, 3.90, 2.23, 2.32, 2.48, 2.43, 3.86, 4.43, 3.78])
mean_speed = np.mean(speed_array)
err_speed = np.std(speed_array, ddof=1) / np.sqrt(len(speed_array))
print(f"\n The mean speed is {mean_speed} +/- {err_speed}")
m_bullet = 8.4 / 1000
m_bullet_err = 0.5 / 1000
E_kin = 1/2 * m_bullet * mean_speed**2
E_kin_err_m_term = (1/2 * mean_speed**2) * m_bullet_err
E_kin_err_speed_term = (m_bullet * mean_speed) * err_speed
E_kin_err = np.sqrt(E_kin_err_m_term**2 + E_kin_err_speed_term**2)
print(f"\n The average kinetic energy is ({E_kin} +/- {E_kin_err})J")
print(f"\n Mass error contribution {E_kin_err_m_term} \n Speed Error contribution {E_kin_err_speed_term}")
err_ratios = E_kin_err_speed_term / E_kin_err_m_term
trials_match_err = err_ratios**2 * len(speed_array)
print(f"\n Number of measurements to match error {trials_match_err:.0f}")
