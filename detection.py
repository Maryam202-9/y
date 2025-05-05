import cv2
import numpy as np

def detect_cells(img_rgb, contours, model):
    output_img = img_rgb.copy()
    count_cells = 0

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w < 20 or h < 20:
            continue

        cell = img_rgb[y:y+h, x:x+w]
        cell_resized = cv2.resize(cell, (64, 64))
        cell_array = np.expand_dims(cell_resized / 255.0, axis=0)

        prediction = model.predict(cell_array, verbose=0)
        class_idx = np.argmax(prediction)

        label = 'RBC' if class_idx == 0 else 'WBC'
        color = (255, 0, 0) if label == 'RBC' else (0, 255, 255)

        cv2.rectangle(output_img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(output_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        count_cells += 1

    return output_img, count_cells
