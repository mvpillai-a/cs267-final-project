import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

# Sample data from your benchmarks
# Replace these with your actual benchmark results
dataset_sizes = [1000, 10000, 100000, 1000000]  # 1K, 10K, 100K, 1M points
cpu_times = [0.563, 1.336, 7.792, 99.850]  # CPU execution times in seconds
cuda_times = [0.036, 0.257, 3.634, 341.417]  # CUDA execution times in seconds

# Create a DataFrame for easier manipulation
df = pd.DataFrame({
    'Dataset Size': dataset_sizes,
    'CPU Time (s)': cpu_times,
    'CUDA Time (s)': cuda_times
})

# Calculate speedup
df['Speedup'] = df['CPU Time (s)'] / df['CUDA Time (s)']

# Create the log-log plot
plt.figure(figsize=(10, 6))

# Plot CPU and CUDA times
plt.loglog(df['Dataset Size'], df['CPU Time (s)'], 'o-', label='CPU Implementation', linewidth=2, markersize=8)
plt.loglog(df['Dataset Size'], df['CUDA Time (s)'], 's-', label='CUDA Implementation', linewidth=2, markersize=8)

# Customize the plot
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.title('NN-Descent Performance: CPU vs. CUDA', fontsize=14)
plt.xlabel('Dataset Size (points)', fontsize=12)
plt.ylabel('Execution Time (seconds)', fontsize=12)

# Set x-axis tick labels
plt.xticks(dataset_sizes, ['1K', '10K', '100K', '1M'])

# Format y-axis to show actual values instead of powers
formatter = ScalarFormatter()
formatter.set_scientific(False)
plt.gca().yaxis.set_major_formatter(formatter)

plt.legend(loc='upper left', fontsize=10)

# Add speedup annotations
for i, row in df.iterrows():
    plt.annotate(f"{row['Speedup']:.2f}x",
                 xy=(row['Dataset Size'], row['CUDA Time (s)']),
                 xytext=(0, -20),
                 textcoords='offset points',
                 ha='center',
                 fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))

# Add a second y-axis for the speedup
ax2 = plt.gca().twinx()
ax2.set_ylabel('Speedup (CPU/CUDA)', color='g', fontsize=12)
ax2.tick_params(axis='y', labelcolor='g')
ax2.set_yscale('log')

# Save the figure
plt.tight_layout()
plt.savefig('nndescent_performance.png', dpi=300)
plt.show()

# Print the data table
print("\nPerformance Data:")
print(df.to_string(index=False))