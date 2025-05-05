import pandas as pd
import cv2
import os

# مسار ملف الـ CSV
annotations_file = 'E:\\FCIH- 4year-end tearm\\computer vision\\cv bllod 2\\archive (7)\\annotations.csv'

# مسار الصور
images_dir = 'E:\\FCIH- 4year-end tearm\\computer vision\\cv bllod 2\\archive (7)\\images'

# مسار الحفظ
output_dir = 'E:/FCIH- 4year-end tearm/computer vision/cv bllod 2/cropped_cells'
rbc_dir = os.path.join(output_dir, 'rbc')
wbc_dir = os.path.join(output_dir, 'wbc')

# إنشاء الفولدرات لو مش موجودة
os.makedirs(rbc_dir, exist_ok=True)
os.makedirs(wbc_dir, exist_ok=True)

# قراءة ملف الannotations
df = pd.read_csv(annotations_file)

# تصحيح الأعمدة لو فيها مسافات
df.columns = df.columns.str.strip()

# نمر على كل annotation
for idx, row in df.iterrows():
    img_name = row['image'].strip()
    xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
    label = row['label'].strip().lower()
    
    img_path = os.path.join(images_dir, img_name)
    
    if not os.path.exists(img_path):
        print(f"الصورة مش موجودة: {img_path}")
        continue
    
    # قراءة الصورة
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"مشكلة بقراءة الصورة: {img_path}")
        continue
    
    # قص الخلية
    cropped = img[ymin:ymax, xmin:xmax]
    
    # تحديد مكان الحفظ حسب نوع الخلية
    if label == 'rbc':
        save_path = os.path.join(rbc_dir, f'{img_name.split(".")[0]}_{idx}.png')
    elif label == 'wbc':
        save_path = os.path.join(wbc_dir, f'{img_name.split(".")[0]}_{idx}.png')
    else:
        continue  # نتجاهل أي label مش rbc أو wbc
    
    # حفظ الصورة
    cv2.imwrite(save_path, cropped)

print("تم قص كل الخلايا وحفظها!")