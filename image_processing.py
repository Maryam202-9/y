import cv2
import numpy as np
import random

def preprocess_image(img_path):
    # 1. قراءة الصورة
    img = cv2.imread(img_path)

    # 2. تغيير الألوان من BGR إلى RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. تصغير الصورة (Resizing)
    img_resized = cv2.resize(img_rgb, (256, 256))

    # 4. تطبيع الصورة (Normalization) [0, 1]
    img_normalized = img_resized / 255.0

    # 5. إزالة الضوضاء (Noise Reduction) باستخدام فلتر Bilateral
    img_denoised = cv2.bilateralFilter((img_normalized * 255).astype('uint8'), d=9, sigmaColor=75, sigmaSpace=75)

    # 6. تحسين التباين (Contrast Adjustment) باستخدام CLAHE
    img_lab = cv2.cvtColor(img_denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    img_clahe = cv2.merge((l_clahe, a, b))
    img_contrast = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2RGB)

    # 7. Augmentation (مثلا تدوير خفيف او قلب الصورة)
    if random.choice([True, False]):
        img_aug = cv2.flip(img_contrast, 1)  # انعكاس أفقي
    else:
        angle = random.choice([-15, -10, 10, 15])
        center = (128, 128)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_aug = cv2.warpAffine(img_contrast, matrix, (256, 256), borderMode=cv2.BORDER_REFLECT)

    # 8. تحويل الصورة إلى رمادية (Gray)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 9. Thresholding
    _, thresh = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    # 10. Blurring (Gaussian Blur)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # 11. Sharpening
    kernel_sharpen = np.array([[0, -1, 0],
                               [-1, 5,-1],
                               [0, -1, 0]])
    img_sharpened = cv2.filter2D(blurred, -1, kernel_sharpen)

    # 12. عمليات مورفولوجية (Morphological Operations)
    kernel = np.ones((3, 3), np.uint8)
    img_morph = cv2.morphologyEx(img_sharpened, cv2.MORPH_OPEN, kernel)
    img_morph = cv2.morphologyEx(img_morph, cv2.MORPH_CLOSE, kernel)

    # في النهاية نرجع الصورة بعد المورفولوجي
    return img_morph,img_rgb, img_gray, thresh
