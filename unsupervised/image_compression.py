from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load image (make sure the image exists in the same folder!)
image = Image.open("image.jpg")  # ✅ CORRECTED
image = image.resize((128, 128))  # Resize to reduce processing

# Show original image
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')
plt.show()

# Convert to numpy array
image_np = np.array(image)

# Get dimensions
w, h, d = image_np.shape
print(f"Image shape: {image_np.shape}")

# Reshape image to 2D array of pixels
pixels = image_np.reshape(-1, 3)

# Set number of colors
n_colors = 4
kmeans = KMeans(n_clusters=n_colors, random_state=42)
kmeans.fit(pixels)

# Replace pixel values with their closest cluster center
compressed_pixels = kmeans.cluster_centers_[kmeans.labels_].astype('uint8')
compressed_image = compressed_pixels.reshape((w, h, d))

# Show compressed image
plt.imshow(compressed_image)
plt.title(f"Compressed Image with {n_colors} colors")
plt.axis('off')
plt.show()
