import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import cv2

from model_loader import load_model
from image_processing import preprocess_image
from segmentation import segment_cells
from feature_extraction import extract_hog_features
from detection import detect_cells
from save_utils import save_combined_image

# تحميل الموديل
model = load_model()

# إعدادات نافذة GUI
root = tk.Tk()
root.title("🔬 Blood Cell Detection")
root.configure(bg="#f0f8ff")  # خلفية مريحة

# ألوان ثابتة
btn_bg = "#4CAF50"
btn_fg = "white"
save_btn_bg = "#1E88E5"
frame_bg = "#e3f2fd"
title_font = ("Helvetica", 10, "bold")
label_font = ("Arial", 12)

top_frame = tk.Frame(root, bg=frame_bg)
top_frame.pack(pady=10)

bottom_frame = tk.Frame(root, bg=root["bg"])
bottom_frame.pack(pady=10)

info_frame = tk.Frame(root, bg=root["bg"])
info_frame.pack(pady=10)

# Progress Bar
progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="indeterminate")
progress.pack(pady=5)
progress.pack_forget()  # نبدأ بإخفائه

def display_images(img_list):
    for widget in top_frame.winfo_children():
        widget.destroy()

    titles = ["Original", "Preprocessing", "Segmentation", "Features", "Detection"]
    for i, img in enumerate(img_list):
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img_pil)

        label_title = tk.Label(top_frame, text=titles[i], font=title_font, bg=frame_bg)
        label_title.grid(row=0, column=i, padx=5, pady=5)

        panel = tk.Label(top_frame, image=img_tk, bg=frame_bg, bd=2, relief="solid")
        panel.image = img_tk
        panel.grid(row=1, column=i, padx=5, pady=5)

def upload_image():
    global final_result
    file_path = filedialog.askopenfilename()
    if file_path:
        progress.pack()
        progress.start()

        root.after(100, lambda: process_image(file_path))  # عشان GUI ما يهنجش

def process_image(file_path):
    global final_result
    try:
        img_morph, img_rgb, img_gray, img_thresh = preprocess_image(file_path)
        contours = segment_cells(img_thresh)
        hog_features, hog_img = extract_hog_features(img_gray)
        detection_img, count_cells = detect_cells(img_rgb, contours, model)

        # Resize Images
        img_resized = cv2.resize(img_rgb, (256, 256))
        thresh_rgb = cv2.cvtColor(img_morph, cv2.COLOR_GRAY2RGB)
        thresh_resized = cv2.resize(thresh_rgb, (256, 256))
        segmentation_img = img_rgb.copy()
        cv2.drawContours(segmentation_img, contours, -1, (0, 255, 0), 2)
        segmentation_resized = cv2.resize(segmentation_img, (256, 256))
        hog_resized = cv2.resize(hog_img, (256, 256))
        detection_resized = cv2.resize(detection_img, (256, 256))

        images = [img_resized, thresh_resized, segmentation_resized, hog_resized, detection_resized]

        display_images(images)

        cells_count_label.config(text=f"عدد الخلايا المكتشفة: {count_cells}")
        feature_text.delete(1.0, tk.END)
        feature_text.insert(tk.END, np.array2string(hog_features[:20], precision=3, separator=', '))
        final_result = images

    finally:
        progress.stop()
        progress.pack_forget()

def save_result():
    if final_result is not None:
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if save_path:
            save_combined_image(final_result, save_path)
            messagebox.showinfo("✅ Saved", "تم حفظ الصورة بنجاح!")

# الأزرار
upload_btn = tk.Button(bottom_frame, text="📁 Upload Image", font=label_font, bg=btn_bg, fg=btn_fg, padx=12, pady=6, command=upload_image)
upload_btn.pack(side=tk.LEFT, padx=10)

save_btn = tk.Button(bottom_frame, text="💾 Save Result", font=label_font, bg=save_btn_bg, fg=btn_fg, padx=12, pady=6, command=save_result)
save_btn.pack(side=tk.LEFT, padx=10)

# عدد الخلايا وخصائص HOG
cells_count_label = tk.Label()
cells_count_label.pack(pady=5)

feature_label = tk.Label(info_frame, text="أول 20 خاصية HOG:", font=label_font, bg=root["bg"], fg="#333")
feature_label.pack()

feature_text = tk.Text(info_frame, height=5, width=100, font=("Courier", 10), bg="#ffffff", fg="#222")
feature_text.pack(pady=5)

final_result = None
root.mainloop()
