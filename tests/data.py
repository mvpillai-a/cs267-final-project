import numpy as np

# Number of points and features
num_points = 100000
num_features = 2

# Generate random data (random floats between 0 and 100)
data = np.random.rand(num_points, num_features) * 100

# Save to a CSV or plain text file
np.savetxt("100k_dataset.txt", data, delimiter=" ", fmt="%.6f")
