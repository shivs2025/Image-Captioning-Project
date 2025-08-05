import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load processed captions
with open('processed_captions.pkl', 'rb') as f:
    embedded_mapping = pickle.load(f)

# Load caption properties
with open('captions_properties.pkl', 'rb') as f:
    caption_prop = pickle.load(f)

max_length = caption_prop['max length']
vocab_size = caption_prop['vocab size']

# Load image features
image_features = np.load('flickr8k_image_features.npy', allow_pickle=True).item()

X1, X2, y = [], [], []

for img_id, captions in embedded_mapping.items():
    img_file = img_id + '.jpg'  # Add extension
    if img_file not in image_features:
        continue
    feature = image_features[img_file]
    for caption_seq in captions:
        for i in range(1, len(caption_seq)):
            in_seq = caption_seq[:i]
            out_seq = caption_seq[i]
            if out_seq == 0:  # skip padding token target
                continue
            in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)

X1 = np.array(X1)
X2 = np.array(X2)
y = np.array(y, dtype='int32')

np.save('X1.npy', X1)
np.save('X2.npy', X2)
np.save('y.npy', y)

print(f"Saved sequences: X1 shape {X1.shape}, X2 shape {X2.shape}, y shape {y.shape}, max_length: {max_length}")
