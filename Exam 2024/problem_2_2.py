import numpy as np
import matplotlib.pyplot as plt

def A_err_term(t, gamma, omega, s_A):
    return np.abs(np.exp(-gamma * t) * np.cos(omega * t) * s_A)

def gamma_err_term(t, A, gamma, omega, s_gamma):
    return np.abs(- A * t * np.exp(-gamma * t) * np.cos(omega * t) * s_gamma)

def omega_err_term(t, A, gamma, omega, s_omega):
    return np.abs(-A * np.exp(-gamma * t) * t * np.sin(omega * t) * s_omega)

def x_err_tot(t, A, gamma, omega, s_A, s_gamma, s_omega):
    A_err = A_err_term(t, gamma, omega, s_A)
    gamma_err = gamma_err_term(t, A, gamma, omega, s_gamma)
    omega_err = omega_err_term(t, A, gamma, omega, s_omega)
    return np.sqrt(A_err**2 + gamma_err**2 + omega_err**2)

t = 1
A = 1.01
s_A = 0.19

gamma = 0.12
s_gamma = 0.05

omega = 0.47
s_omega = 0.06

x_err = x_err_tot(t, A, gamma, omega, s_A, s_gamma, s_omega)

print(f"\n The total error on x is: {x_err:.3f}")

t_array = np.linspace(0, 60, 10_000)
A_err_array = A_err_term(t_array, gamma, omega, s_A)
gamma_err_array = gamma_err_term(t_array, A, gamma, omega, s_gamma)
omega_err_array = omega_err_term(t_array, A, gamma, omega, s_omega)

# fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
# axs[0].plot(t_array, A_err_array, label=r'$A$', color='red')
# axs[1].plot(t_array, gamma_err_array, label=r'$\gamma$', color='blue')
# axs[2].plot(t_array, omega_err_array, label=r'$\omega$', color='black')
# axs[1].set_xlabel("t", fontsize=15)
# axs[0].set_ylabel(r"$\sigma_x$ contribution", fontsize=15)
# axs[0].legend(fontsize=15)
# axs[1].legend(fontsize=15)
# axs[2].legend(fontsize=15)
# plt.show()
plt.figure(figsize=(10, 5))
plt.plot(t_array, A_err_array, label=r'$A$', color='red', lw=2, alpha=1, zorder=1)
plt.plot(t_array, gamma_err_array, label=r'$\gamma$', color='blue', lw=2, alpha=0.7)
plt.plot(t_array, omega_err_array, label=r'$\omega$', color='black', lw=2, alpha=0.3)
plt.xlabel("t", fontsize=15)
plt.ylabel(r"$\sigma_x$ contribution", fontsize=15)
plt.legend(fontsize=15)
# plt.savefig(r"/home/ali/Figures/fig1.png", bbox_inches='tight', dpi=500)
plt.show()
