import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# First curve data
x1 = [0.21, 0.34, 0.0011892044771, 0.15, 0.18, 0.0028188550568, 0.0037584734091, 0.0050112978788, 0.0066817305051, 0.0089089740068]
y1 = [1e-06, 1e-06, 0.0, 1e-06, 0.0, 4e-06, 1.3e-05, 1e-05, 2.4e-05]

# Second curve data
x2 = [0.0008919033578, 0.0011892044771, 0.0015856059695, 0.0021141412926, 0.0028188550568, 0.0037584734091, 0.0050112978788, 0.0066817305051, 0.0089089740068]
y2 = [2e-06, 2e-06, 1e-06, 2e-06, 1e-06, 5e-06, 1.4e-05, 1.1e-05, 2.5e-05]

correlation_coefficient = np.corrcoef(y1, y2)[0, 1]
print("Correlation coefficient:", correlation_coefficient)
# Calculate Pearson correlation coefficient between the two curves
pearson_corr, p_value = pearsonr(x1, x2)
print("Pearson correlation coefficient between the two curves:", pearson_corr)
print("P-value:", p_value)

# Plotting both curves
plt.plot(x1, y1, label='Curve 1')
plt.plot(x2, y2, label='Curve 2')

# Adding labels and legend
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Comparison of Two Curves')
plt.legend()

# Displaying the plot
plt.show()



