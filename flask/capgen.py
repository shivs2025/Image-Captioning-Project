
import os
# Set an environment variable
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from PIL import Image
import pickle
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.saving import load_model
 
app = Flask(__name__)
 
upload_folder = os.path.join('static', 'uploads')
 
app.config['UPLOAD'] = upload_folder

t = pickle.load(open('tokenizer.pkl','rb'))
vocab_size = len(t.word_index) + 1
maxlen = 34

model = load_model("model_13.keras")

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    #add start tag
    in_text = 'start'
    #iterate over the max length of sequence
    for i in range(max_length):
        #encode the input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        #pad the sequence
        sequence = pad_sequences([sequence],max_length,padding='post')
        #predict next word
        yhat = model.predict([image, sequence], verbose=0)
        #get index with highest probability
        yhat = np.argmax(yhat)
        #convert index to word
        word = idx_to_word(yhat,tokenizer)
        #stop if word not found
        if word is None:
            break
        #append word as input for generating next word
        in_text += ' ' + word
        #stop if we have reached the end tag
        if word == 'end':
            break
    return in_text

def get_feature(img_path):
    mv3 = InceptionV3()
    mv3 = Model(inputs=mv3.inputs, outputs=mv3.layers[-2].output)
    image = load_img(img_path,target_size=(299,299))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = image / 255
    feature = mv3.predict(image, verbose=0)
    return feature

def generate_caption(image_name):
    #load the image
    image_id = image_name.split('.')[0]
    img_path = os.path.join(upload_folder,image_name)
    image = Image.open(img_path)
    #predict the caption
    image_feature = get_feature(img_path)
    y_pred = predict_caption(model, image_feature, t, maxlen)
    return y_pred
 
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img = os.path.join(app.config['UPLOAD'], filename)
        caption = generate_caption(filename)
        #get rid of start and end
        caption = ' '.join(caption.split()[1:-1])
        return render_template('image_render.html', img=img, cap=caption)
    return render_template('image_render.html')
  
if __name__ == '__main__':
    app.run(debug=True)