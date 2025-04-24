import numpy as np

# Generate 10 2D float32 vectors
data = np.random.rand(10, 2).astype(np.float32)

# Save as ParlayANN-compatible fbin format: [int32 count][float32 data...]
with open("data_10pts_2D.fbin", "wb") as f:
    f.write(np.array([10, 2], dtype=np.int32).tobytes())  # Header: 10 points, 2 dimensions
    f.write(data.tobytes())

