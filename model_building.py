import numpy as np
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.callbacks import ModelCheckpoint

# Load training data
X1 = np.load('X1.npy')
X2 = np.load('X2.npy')
y = np.load('y.npy')

# Load caption properties
with open('captions_properties.pkl', 'rb') as f:
    caption_prop = pickle.load(f)
vocab_size = caption_prop['vocab size']
max_length = caption_prop['max length']

# Model architecture
inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)

inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model.summary()

checkpoint = ModelCheckpoint('model_caption.h5', monitor='loss', save_best_only=True, verbose=1)
model.fit([X1, X2], y, epochs=20, batch_size=256, callbacks=[checkpoint])

model.save('model_caption.keras')
