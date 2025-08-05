from keras.models import load_model
import numpy as np
import pickle
from keras.preprocessing.sequence import pad_sequences
from PIL import Image
import matplotlib.pyplot as plt
from caption_process import maxlen

# Set your max_length as in your data pipeline, or load it
max_length = maxlen  

# Load model
model = load_model('model_caption.keras', compile=False)

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
index_word = {v: k for k, v in tokenizer.word_index.items()}

def generate_caption(photo, max_len=max_length):
    in_text = 'start'
    for _ in range(max_len):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_len)
        yhat = model.predict([photo, seq], verbose=0)
        predicted_id = np.argmax(yhat)
        word = index_word.get(predicted_id)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return ' '.join(in_text.split()[1:-1])

def display_image_with_caption(image_path, caption):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(caption, fontsize=12)
    plt.show()

# Load image features from npy file
image_features = np.load('flickr8k_image_features.npy', allow_pickle=True).item()

# --- Choose an image ---
image_path = 'C:/Users/swapn/Downloads/archive/Images/95728664_06c43b90f1.jpg' #Your image directory
image_id = image_path.split("/")[-1] 

if image_id not in image_features:
    raise ValueError(f"Image ID {image_id} not found in features.")

photo = image_features[image_id].reshape((1, 2048))
caption = generate_caption(photo)

print(f"🖼️ Image ID: {image_id}")
print(f"📜 Caption: {caption}")

display_image_with_caption(image_path, caption)
