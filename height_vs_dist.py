import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load the data
data = np.loadtxt('dist.dat')
height = data[:, 0]  # 1st column: height in cm
distance = data[:, 1]  # 2nd column: distance in meters
std_dev = data[:, 2]  # 3rd column: standard deviation

# Perform linear regression
slope, intercept, r_value, _, _ = linregress(height, distance)

# Create the plot
plt.errorbar(height, distance, yerr=std_dev, fmt='o', label='Data')
plt.plot(height, intercept + slope * height, 'r', label=f'Fit: y={slope:.2f}x+{intercept:.2f}\n$R^2$={r_value**2:.2f}')

# Adding labels and title
plt.xlabel('Height (cm)')
plt.ylabel('Distance (m)')
plt.title('Linear Regression of Distance vs Height')
plt.legend()

# Save the figure in EPS format
plt.savefig('linear_regression_plot.eps', format='eps')

# Show the plot
plt.show()
