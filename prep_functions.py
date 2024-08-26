import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class ImagePreprocessor:
    def __init__(self):
        pass

    def save_image(self, image, path, name):
        # Save the image to the specified path
        result_image = Image.fromarray(np.uint8(image))
        save_path = path + name
        result_image.save(save_path, format='PNG')

    def resize_image(self, image, width, height):
        # Resize the image to the specified dimensions
        image = cv2.resize(image, (width, height))
        image = image / 255.0  # Normalize to [0, 1]
        return image

    def show_image(self, image, name):
        # Display the image using OpenCV
        cv2.imshow(name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def threshold_image(self, image, threshold_value):
        # Ensure threshold_value is a scalar
        lower = threshold_value
        upper = 255  # Assuming 8-bit image
        thresh = cv2.inRange(image, lower, upper)
        return thresh

    def apply_morphology(self, img, thresh):
        # Apply morphology operation on the image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # invert morph image
        mask = 255 - morph

        # apply mask to image
        result = cv2.bitwise_and(img, img, mask=mask)
        return result

    def clahe_he(self, image):
        # Apply histogram equalization
        equalized_image = cv2.equalizeHist(image)

        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=300, tileGridSize=(8, 8))

        # Apply CLAHE
        clahe_image = clahe.apply(equalized_image)

        return clahe_image

    def histogram_equalization(self, image):
        # Enhance the contrast of the image using histogram equalization.
        # This method redistributes the pixel intensity values to span the entire range.
        return cv2.equalizeHist(image)

    def gaussian_smoothing(self, image, kernel_size=5):
        # Apply Gaussian smoothing to reduce noise and detail in the image.
        # The kernel size controls the amount of blurring applied.
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def apply_threshold(self, image, threshold=128):
        # Apply a binary threshold to the image.
        # Pixels with intensity values above the threshold are set to 255 (white),
        # while pixels with values below the threshold are set to 0 (black).
        _, thresh_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return thresh_image

    def otsu_thresholding(self, image):
        # Apply Otsu's thresholding method to automatically determine the optimal threshold value.
        # This method assumes a bimodal distribution of pixel intensities.
        _, thresh_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh_image

    def roi_extraction(self, image, thresh_image):
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Create an empty mask to draw the contours
        mask = np.zeros_like(image)

        # Draw the contours on the mask
        cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
        # Invert the mask to keep only the ROI
        mask = cv2.bitwise_not(mask)

        # Apply the mask to the original image
        result = cv2.bitwise_and(image, image, mask=mask)
        return result

    def brightness_adjustment(self, image_path, alpha, beta):
        # Implement image brightness adjustment logic
        # read the input image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # call convertScaleAbs function
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted

    def sharpen_image(self, image_path):
        # Load image
        image = cv2.imread(image_path)

        # Create sharpening kernel
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

        # Apply kernel to image
        sharpened_image = cv2.filter2D(image, -1, kernel)

        return sharpened_image

    def denoise_image(self, image):
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

    def normalize_image(self, image):
        # Normalize the pixel values to the range [0, 1].
        # This is useful for models that require input in this range.
        normalized_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return normalized_image

    def rotate_image(self, image, angle):
        # Rotate the image around its center by a given angle.
        # The angle is in degrees and positive values mean counter-clockwise rotation.

        # Get the image dimensions
        (h, w) = image.shape[:2]
        # Calculate the center of the image
        center = (w // 2, h // 2)
        # Create the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        # Perform the rotation
        rotated_image = cv2.warpAffine(image, M, (w, h))
        return rotated_image
