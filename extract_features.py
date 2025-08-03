import os
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tqdm import tqdm

# Step 1: Define image folder path
IMAGE_DIR = 'Images'

# Step 2: Load InceptionV3 model
print("Loading InceptionV3 model...")
base_model = InceptionV3(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
TARGET_SIZE = (299, 299)

# Step 3: Feature Extraction Function
def extract_features(image_dir):
    features = {}
    for img_name in tqdm(os.listdir(image_dir), desc="Extracting features"):
        img_path = os.path.join(image_dir, img_name)

        try:
            img = image.load_img(img_path, target_size=TARGET_SIZE)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            feature = model.predict(x, verbose=0)
            features[img_name] = feature.flatten()
        
        except Exception as e:
            print(f"Error with image {img_name}: {e}")
    return features

# Step 4: Run and Save
if __name__ == "__main__":
    print("Starting feature extraction...")
    features = extract_features(IMAGE_DIR)
    
    print("Saving to flickr8k_image_features.npy...")
    np.save('flickr8k_image_features.npy', features)

    print("Done! Features saved successfully.")
