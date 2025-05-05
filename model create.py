import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

# مسار الخلايا المقصوصة
dataset_dir = 'E:/FCIH- 4year-end tearm/computer vision/cv bllod 2/cropped_cells'

# تجهيز البيانات
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # مهم علشان المؤشرات متتلغبطش
)

# بناء موديل CNN بسيط
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')  # rbc / wbc
])

# كومبايل
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# تدريب
model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# تقييم النموذج بالتفصيل
y_true = []
y_pred = []

# تمرير كل الصور في validation set للحصول على توقعات النموذج
for images, labels in validation_generator:
    preds = model.predict(images)
    y_true.extend(np.argmax(labels, axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

    if len(y_true) >= validation_generator.samples:
        break

# طباعة المقاييس
target_names = list(validation_generator.class_indices.keys())
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=target_names))

# حفظ الموديل (اختياري)
# model.save('blood_cnn_model.h5')

print("تم تدريب وتقييم النموذج!")
