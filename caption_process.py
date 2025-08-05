import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import contractions

# Text preprocessing
def clean_text(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()
            caption = contractions.fix(caption)
            caption = re.sub(r'[^a-z\s]', '', caption)
            caption = re.sub(r'\s+', ' ', caption)
            caption = ' '.join([word for word in caption.split() if len(word) > 1])
            caption = 'start ' + caption + ' end'
            captions[i] = caption
    return

# Load captions
with open('captions.txt', 'r') as f:
    next(f)
    captions_doc = f.read()
print('Captions.txt loaded successfully..')

# Create mapping of image to captions
mapping = {}
for lines in captions_doc.split('\n'):
    tokens = lines.split(',')
    if len(tokens) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    image_id = image_id.split('.')[0]  # remove file extension
    caption = ' '.join(caption)
    if image_id not in mapping:
        mapping[image_id] = []
    mapping[image_id].append(caption)

# Preprocess text
clean_text(mapping)
print('Cleaning text and putting tag complete..')

# Create tokens for caption
cleaned_corpus = []
for key in mapping:
    for caption in mapping[key]:
        cleaned_corpus.append(caption)

maxlen = max(len(caption.split()) for caption in cleaned_corpus)

# Assign indexes to unique words
t = Tokenizer()
t.fit_on_texts(cleaned_corpus)
vocab_size = len(t.word_index) + 1

# Save tokenizer for use in caption generation
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(t, f)

# Tokenize and pad captions
embedded_mapping = {}
for key, captions in mapping.items():
    for i in range(len(captions)):
        tokenized_caption = t.texts_to_sequences([captions[i]])[0]
        padded_tokenized_caption = pad_sequences([tokenized_caption], maxlen=maxlen, padding='post')[0]
        if key not in embedded_mapping:
            embedded_mapping[key] = []
        embedded_mapping[key].append(padded_tokenized_caption)

# Save processed captions and properties
pickle.dump(embedded_mapping, open('./processed_captions.pkl', 'wb'))

caption_prop = {
    'vocab size': vocab_size,
    'max length': maxlen
}
pickle.dump(caption_prop, open('./captions_properties.pkl', 'wb'))

