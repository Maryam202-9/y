import cv2

def save_combined_image(images_list, save_path):
    combined = cv2.hconcat(images_list)
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, combined_bgr)
