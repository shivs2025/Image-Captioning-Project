import os
# Set an environment variable
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import contractions

#Text preprocessing
def clean_text(mapping):
    for key,captions in mapping.items():
        for i in range(len(captions)):
            # tke one caption at a time
            caption = captions[i]
            # preprocessing steps
            # convert to lower case 
            caption = caption.lower()
            # expand contractions
            caption = contractions.fix(caption)
            # delete special charectors, digits and others
            #caption = caption.replace('[^a-z]','')
            caption = re.sub(r'[^a-z\s]','',caption)
            # replacing multiple space with one space
            #caption = caption.replace('\s+',' ')
            caption = re.sub(r'\s+',' ',caption)
            # remove one charector word
            caption = ' '.join([word for word in caption.split() if len(word)>1])
            # add start and end
            caption = 'start ' + caption + ' end'
            captions[i] = caption
    return

# load captions
with open('./captions.txt','r') as f:
    next(f)
    captions_doc = f.read()
print('Captions.txt loaded successfully..')

#create mapping of image to captions
mapping = {}
#process lines
for lines in captions_doc.split('\n'):
    #split the line by comma
    tokens = lines.split(',')
    if len(tokens) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    #remove file extension from image_id
    image_id = image_id.split('.')[0]
    #convert caption list to string
    caption = ' '.join(caption)
    #create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    #store the caption
    mapping[image_id].append(caption) 

#preprocess the text
clean_text(mapping)

print('Cleaning text and putting <start> <end> tag complete..')

#create tokens for caption
cleaned_corpus = []
for key in mapping:
    for caption in mapping[key]:
        cleaned_corpus.append(caption)
        
# get the maximum length of the captions available
maxlen = max(len(caption.split()) for caption in cleaned_corpus)

#assign indexes to unique words in corpus
t = Tokenizer()
t.fit_on_texts(cleaned_corpus)
vocab_size = len(t.word_index) + 1

# Tokenize and pad the caption
embedded_mapping ={}
for key,captions in mapping.items():
    for i in range(len(captions)):
        tokenized_caption = t.texts_to_sequences([captions[i]])[0]
        padded_tokenized_caption = pad_sequences([tokenized_caption],maxlen,padding='post')[0]    #0 to eliminate outside []
        #create list if needed
        if key not in embedded_mapping:
            embedded_mapping[key] = []
        #store the caption
        embedded_mapping[key].append(padded_tokenized_caption)

# store captions for future use in the pipeline
pickle.dump(embedded_mapping,open('./processed_captions.pkl','wb'))

# store values for future use in the pipeline
caption_prop = {}
caption_prop['vocab size'] = vocab_size
caption_prop['max length'] = maxlen
pickle.dump(caption_prop,open('./captions_properties.pkl','wb'))

print('Captions processing complete. Check for pickle files processed_captions, captions_properties.')

