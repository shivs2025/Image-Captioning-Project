# ==========================================================
# 1. Dataset Handling (Flickr8k version of MS COCO step)
# ==========================================================

import os
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np

# Paths
IMAGE_FOLDER = os.path.join("data", "raw", "images")
PROCESSED_IMAGE_DIR = os.path.join("data", "processed", "resized_images")

# Create processed image folder (if saving)
os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)

# Preprocess and save resized/normalized image arrays
def preprocess_and_save_images(image_dir, target_dir):
    for image_name in tqdm(os.listdir(image_dir)):
        img_path = os.path.join(image_dir, image_name)
        try:
            img = load_img(img_path, target_size=(299, 299))
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)  # Normalize for InceptionV3
            # Optional: Save image as .npy
            np.save(os.path.join(target_dir, image_name.split('.')[0] + '.npy'), img_array)
        except Exception as e:
            print(f"Failed to process {image_name}: {e}")

# Run it
preprocess_and_save_images(IMAGE_FOLDER, PROCESSED_IMAGE_DIR)
print("All images resized and normalized.")