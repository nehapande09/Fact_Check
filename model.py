# model.py
import pickle
from tensorflow.keras.models import load_model

def load_model_and_tokenizer():
    model = load_model("rnn_model.h5")
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer
