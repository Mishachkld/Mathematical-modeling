import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import img_as_float
from skimage.transform import radon, iradon

image = imread("img.jpg")
gray_image = rgb2gray(image)
# нормализуем
normalized_image = img_as_float(gray_image)

angels = [1, 5, 10]
invert_radons = []

for angle in angels:
    theta = np.arange(0, 180, angle)
    sinogram = radon(normalized_image, theta, circle=True)
    invert_radon = iradon(sinogram, theta, circle=True)
    invert_radons.append((angle, sinogram, invert_radon))

fig, axes = plt.subplots(3, 3, figsize=(9, 9))

for i in range(len(invert_radons)):
    angle, sinogram, invert_radon = invert_radons[i]
    axes[i][0].imshow(normalized_image, cmap='gray')

    axes[i][1].imshow(sinogram, cmap='gray', aspect="auto")
    axes[i][1].set_title(f"Angle {angle}")

    axes[i][2].imshow(invert_radon, cmap='gray')
    for j in range(3):
        axes[i][j].axis('off')

plt.tight_layout()
plt.show()