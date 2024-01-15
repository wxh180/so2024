import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the data
data = np.loadtxt('angle.dat')
angle = data[:, 0]
distance = data[:, 1]
std_dev = data[:, 2]

# Define the models
def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def exponential(x, a, b):
    return a * np.exp(b * x)

def logarithmic(x, a, b):
    return a * np.log(x) + b

def power_law(x, a, b):
    return a * x**b

# Fit the models
models = [quadratic, exponential, logarithmic, power_law]
model_names = ["Quadratic", "Exponential", "Logarithmic", "Power Law"]
best_r_squared = -np.inf
best_model_func = None
best_params = None

for model, name in zip(models, model_names):
    try:
        params, _ = curve_fit(model, angle, distance, maxfev=10000)
        predicted = model(angle, *params)
        ss_res = np.sum((distance - predicted) ** 2)
        ss_tot = np.sum((distance - np.mean(distance)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        if r_squared > best_r_squared:
            best_r_squared = r_squared
            best_model_func = model
            best_model_name = name
            best_params = params
    except:
        print(f"Error fitting {name} model")

# Function to format the equation string
def format_equation(params, model_name):
    if model_name == "Quadratic":
        return f"{params[0]:.5f}x\u00b2 + {params[1]:.5f}x + {params[2]:.5f}"
    elif model_name == "Exponential":
        return f"{params[0]:.5f}e^({params[1]:.5f}x)"
    elif model_name == "Logarithmic":
        return f"{params[0]:.5f}ln(x) + {params[1]:.5f}"
    elif model_name == "Power Law":
        return f"{params[0]:.5f}x^{params[1]:.5f}"
    else:
        return "Unknown Model"

# Create the plot
plt.errorbar(angle, distance, yerr=std_dev, fmt='o', label='Data')

# Plot the best fitting model
predicted_distance = best_model_func(angle, *best_params)
equation = format_equation(best_params, best_model_name)
plt.plot(angle, predicted_distance, 'r', label=f'Best Fit: {best_model_name}\nEquation: {equation}\n$R^2$={best_r_squared:.5f}')

# Adding labels and title
plt.xlabel('Angle (degrees)')
plt.ylabel('Distance (m)')
plt.title('Best Non-linear Fit')

plt.legend(loc='upper left',bbox_to_anchor=(1,1))

plt.savefig('best_non_linear_fit.eps', format='eps', bbox_inches='tight')
plt.show()
