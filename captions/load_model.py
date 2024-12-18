from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import DenseNet201
import pickle

def get_model():
    model = load_model(r'models/image_caption.h5')
    return model

def get_fe_model():
    fe_model = DenseNet201(weights='imagenet')
    model_new = Model(fe_model.input, fe_model.layers[-2].output)
    return model_new


def get_tokenizer():
    with open(r'models/tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer
