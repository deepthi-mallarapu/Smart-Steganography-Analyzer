from PIL import Image
import numpy as np

# Create a 256x256 grayscale gradient image
width, height = 256, 256
array = np.tile(np.arange(256, dtype=np.uint8), (256, 1))  # gradient 0-255

img = Image.fromarray(array)
img.save("static/uploads/sample.png")  # save in uploads folder

print("Sample image created at static/uploads/sample.png")
