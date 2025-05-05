from skimage.feature import hog
import cv2

def extract_hog_features(gray_img):
    features, hog_image = hog(
        gray_img,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True
    )
    hog_image_normalized = (hog_image / hog_image.max() * 255).astype('uint8')
    return features, cv2.cvtColor(hog_image_normalized, cv2.COLOR_GRAY2RGB)
