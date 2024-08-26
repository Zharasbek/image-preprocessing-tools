import cv2
from prep_functions import ImagePreprocessor
import numpy as np


# Create an instance of the ImagePreprocessor class
preprocessor = ImagePreprocessor()

dataset_path = 'img/'
image_path = dataset_path + 'p5.png'

# Load the grayscale image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Test histogram equalization
he_image = preprocessor.histogram_equalization(image)
preprocessor.show_image(he_image, 'Histogram Equalization')

# Test Gaussian smoothing
blurred = preprocessor.gaussian_smoothing(he_image)
preprocessor.show_image(blurred, 'Gaussian Smoothing')

# Test thresholding
thresholded_image = preprocessor.apply_threshold(blurred)
preprocessor.show_image(thresholded_image, 'Thresholded Image')

# Test Otsu thresholding
otsu_image = preprocessor.otsu_thresholding(blurred)
preprocessor.show_image(otsu_image, 'Otsu Thresholding')

# Test image denoising
denoised_image = preprocessor.denoise_image(image)
preprocessor.show_image(denoised_image, 'Denoised Image')

# Test image normalization
normalized_image = preprocessor.normalize_image(image)
preprocessor.show_image(normalized_image, 'Normalized Image')

# Test image rotation
rotated_image = preprocessor.rotate_image(image, 45)  # Rotate by 45 degrees
preprocessor.show_image(rotated_image, 'Rotated Image')

# Test edge detection using Canny
edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
preprocessor.show_image(edges, 'Canny Edges')

# Test contour detection
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)
preprocessor.show_image(image_contours, 'Contours')

# Test saving an image
# preprocessor.save_image(image, dataset_path, 'image_name.png')

