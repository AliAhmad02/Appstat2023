import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.stats import normaltest

lower = 0.005
upper = 1

int_non_norm = quad(lambda x: x**(-0.9), lower, upper)[0]
C = 1 / int_non_norm
print(f"\n Normalization constant C: {C:.3f}")

print(f"\n We use accept-reject as it is bounded in x and y")
print(f"\n If it was [0,1], we could use transformation, as it would not be bounded in y.")

x_lower = 0.005
x_upper = 1

y_lower = C * x_upper**(-0.9)
y_upper = C * x_lower**(-0.9)

x_vals = np.random.uniform(x_lower, x_upper, 100_000)
y_vals = np.random.uniform(y_lower, y_upper, 100_000)
f_vals = C * x_vals**(-0.9)
accepted_vals = x_vals[y_vals < f_vals]

plt.figure(figsize=(6,4))
plt.hist(accepted_vals, bins=100, histtype='step')
plt.show()

t_vals = []
N = 1000

for i in range(N):
    t_vals.append(
        np.sum(
            np.random.choice(accepted_vals, size=50)
        )
    )
plt.figure(figsize=(6,4))
plt.hist(t_vals, bins=80, histtype='step')
plt.show()

t_mean = np.mean(t_vals)
t_err = np.std(t_vals, ddof=1) / np.sqrt(len(t_vals))
print(f"Mean value of t {t_mean} +/- {t_err}")
p_val_normaltest = normaltest(t_vals).pvalue
print(f"\n The p-value for the normal test {p_val_normaltest}")