import pickle
import joblib  # For loading ML models
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_label(model, tokenizer, text, max_length=100):
    """Predicts the label using the RNN model."""
    try:
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=max_length, padding="post")

        prediction = model.predict(padded_sequence)
        confidence = float(prediction[0][0]) * 100

        if confidence > 50:
            return "True", confidence
        else:
            return "False", 100 - confidence

    except Exception as e:
        print(f"❌ Error in predict_label: {e}")
        return "Error", 0

def load_rnn_model_and_tokenizer():
    """Loads the trained RNN model and tokenizer."""
    try:
        print("🔄 Loading RNN model...")
        model = load_model("rnn_model.h5")
        print("✅ RNN Model loaded successfully.")

        print("🔄 Loading tokenizer...")
        with open("tokenizer.pickle", "rb") as handle:
            tokenizer = pickle.load(handle)
        print("✅ Tokenizer loaded successfully.")

        return model, tokenizer

    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
    except Exception as e:
        print(f"❌ Error loading RNN model or tokenizer: {e}")

    return None, None

def load_ml_model():
    """Loads the trained ML model and TF-IDF vectorizer."""
    try:
        print("🔄 Loading ML model...")
        with open("ml_model.pkl", "rb") as model_file:
            ml_model = pickle.load(model_file)

        print("🔄 Loading TF-IDF vectorizer...")
        with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)

        print("✅ ML Model and Vectorizer loaded successfully!")
        return ml_model, vectorizer

    except FileNotFoundError:
        print("❌ ML Model or Vectorizer file not found!")
    except Exception as e:
        print(f"❌ Error loading ML Model: {e}")

    return None, None

if __name__ == "__main__":
    rnn_model, tokenizer = load_rnn_model_and_tokenizer()
    ml_model, vectorizer = load_ml_model()

    if rnn_model and tokenizer:
        print("✅ RNN Model and Tokenizer loaded correctly!")
    else:
        print("❌ Failed to load RNN Model or Tokenizer.")

    if ml_model and vectorizer:
        print("✅ ML Model and Vectorizer loaded correctly!")
    else:
        print("❌ Failed to load ML Model or Vectorizer.")

# ✅ Make functions available for import
__all__ = ["load_rnn_model_and_tokenizer", "predict_label", "load_ml_model"]
