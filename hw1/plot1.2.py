import matplotlib.pyplot as plt

# XOR-like dataset
X = [(0, 0), (0, 1), (1, 0), (1, 1)]
y = [0, 1, 1, 0]

# Separate points by class for coloring
x0 = [X[i][0] for i in range(len(X)) if y[i] == 0]
y0 = [X[i][1] for i in range(len(X)) if y[i] == 0]

x1 = [X[i][0] for i in range(len(X)) if y[i] == 1]
y1 = [X[i][1] for i in range(len(X)) if y[i] == 1]

# Plot
plt.scatter(x0, y0, color='blue', marker='o', label='y = 0')
plt.scatter(x1, y1, color='orange', marker='s', label='y = 1')

plt.title("XOR Training Set — Greediness Example")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.xticks([0, 1])
plt.yticks([0, 1])
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

plt.savefig("xor_plot.png", dpi=300)
plt.show()
