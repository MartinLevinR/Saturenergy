import numpy as np
import matplotlib.pyplot as plt

def generate_gaussian_samples(mean, std_dev, num_samples):
    samples = []
    for _ in range(num_samples // 2):
        U1, U2 = np.random.rand(2)  # Generate two independent uniform random numbers
        Z0 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
        Z1 = np.sqrt(-2 * np.log(U1)) * np.sin(2 * np.pi * U2)
        samples.append(Z0 * std_dev + mean)
        samples.append(Z1 * std_dev + mean)
    # If num_samples is odd, we need one more sample
    if num_samples % 2 != 0:
        U1, U2 = np.random.rand(2)
        Z0 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
        samples.append(Z0 * std_dev + mean)
    return np.array(samples)

# Parameters for the Gaussian distribution
mean = 100
std_dev = 5
num_samples = 10000

# Generate samples
samples = generate_gaussian_samples(mean, std_dev, num_samples)

# Plot the histogram of the generated samples
plt.hist(samples, bins=50, density=True, alpha=0.6, color='g')

# Plot the theoretical Gaussian distribution
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = (1 / (np.sqrt(2 * np.pi) * std_dev)) * np.exp(-0.5 * ((x - mean) / std_dev)**2)
plt.plot(x, p, 'k', linewidth=2)
title = "Histogram of Generated Gaussian Samples\nMean = {:.2f}, Std Dev = {:.2f}".format(mean, std_dev)
plt.title(title)
plt.show()

cminus = 5
cplus = 10
results = []

for j in np.arange(0.19, 0.55, 0.01):
    newmean = mean + j * std_dev
    pokutaplus = 0
    pokutaminus = 0
    for i in samples:
        if i < newmean:
            pokutaminus += (newmean - i) * cminus
        elif i >= newmean:
            pokutaplus += (i - newmean) * cplus
    pair = (round(j,2), abs(pokutaplus - pokutaminus))
    results.append(pair)
    # print("For newmean =  mean + ", j, " * std_dev the result diff is = ", abs(pokutaplus - pokutaminus))

min_result = min(results, key=lambda x: x[1])
print("Minimum result is for ", min_result[0], "with a value of = ", min_result[1])

precise = []

for g in np.arange(min_result[0]-0.1, min_result[0]+0.1, 0.0005):
    newmean = mean + g * std_dev
    pokutaplus = 0
    pokutaminus = 0
    for i in samples:
        if i < newmean:
            pokutaminus += (newmean - i) * cminus
        elif i >= newmean:
            pokutaplus += (i - newmean) * cplus
    pair = (round(g, 3), abs(pokutaplus - pokutaminus))
    precise.append(pair)

min_precise = min(precise, key=lambda x: x[1])
print("Minimum precise is for ", min_precise[0], "with a value of = ", min_precise[1])


