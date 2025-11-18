import numpy as np
import matplotlib.pyplot as plt

# Number of datasets
T = 10_000
n = 200

errors = np.zeros(T)

for t in range(T):
    # 1. Generate uniform samples
    x = np.random.uniform(-1, 1, size=n)
    y = (x >= 0).astype(int)

    # Find the smallest positive item
    pos = x[y == 1]
    if len(pos) > 0:
        a = np.min(pos)
    else:
        a = 1  # as defined in the homework

    # True risk = a/2 (proven earlier)
    R_fa = a / 2                       # true risk
    R_hat = 0                          # empirical risk is always zero
    
    errors[t] = R_fa - R_hat

# ---- Results ----

# Histogram
plt.hist(errors, bins=40, edgecolor='black')
plt.title("Histogram of R(f_a) - R_hat(f_a) over 10,000 repetitions")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()

# 95% quantile
q95 = np.quantile(errors, 0.95)

print(f"Mean error: {errors.mean():.5f}")
print(f"95% quantile: {q95:.5f}")

# Example PAC bound result from section 1.5
pac_bound = 0.65
print(f"PAC bound from part 1.5: {pac_bound:.5f}")
print(f"Ratio (empirical 95% quantile / PAC bound): {q95/pac_bound:.4f}")
