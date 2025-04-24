import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import subprocess
import os

# Convert DOT to PNG using Graphviz
def dot_to_png(dot_file: str, png_file: str):
    subprocess.run(['dot', '-Tpng', dot_file, '-o', png_file], check=True)

# Display PNG using matplotlib
def show_image(png_file: str):
    img = mpimg.imread(png_file)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Graph Visualization')
    plt.show()

if __name__ == "__main__":
    dot_file = "output_graph.dot"
    png_file = "output_graph.png"
    
    if not os.path.exists(dot_file):
        print(f"{dot_file} not found. Run your C++ code first to generate it.")
    else:
        dot_to_png(dot_file, png_file)
        show_image(png_file)

