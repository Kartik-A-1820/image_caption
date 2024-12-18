from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np
from .load_model import get_fe_model, get_tokenizer, get_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

img_size=224
max_length=74

model_new = get_fe_model()
tokenizer = get_tokenizer()
caption_model = get_model()

def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def feature_extract(img_path):
    img = preprocess_image(img_path, target_size=(img_size, img_size))
    features = model_new.predict(img, verbose=0)
    return features.reshape(1,-1)

def idx_to_word(integer,tokenizer):
    
    for word, index in tokenizer.word_index.items():
        if index==integer:
            return word
    return None

def predict_caption(image_path):
    
    feature = feature_extract(image_path)
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)

        y_pred = caption_model.predict([feature,sequence], verbose=0)
        y_pred = np.argmax(y_pred)
        
        word = idx_to_word(y_pred, tokenizer)
        
        if word is None:
            break
            
        in_text+= " " + word
        
        if word == 'endseq':
            break
    in_text = in_text.replace('startseq', '')
    in_text = in_text.replace('endseq', '')
    return in_text
