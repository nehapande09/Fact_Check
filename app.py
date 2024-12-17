from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Initialize Flask app
app = Flask(__name__)


# Load the trained model and tokenizer
def load_model_and_tokenizer():
    model = load_model("rnn_model.h5")
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer


model, tokenizer = load_model_and_tokenizer()


# Preprocessing function for claims
def predict_label(model, tokenizer, claims):
    sequences = tokenizer.texts_to_sequences(claims)
    max_length = 100  # Ensure consistency with training
    data = pad_sequences(sequences, maxlen=max_length)
    predictions = model.predict(data)
    labels = ["SUPPORTS" if pred > 0.5 else "REFUTES" for pred in predictions]
    confidences = [pred[0] * 100 for pred in predictions]  # Convert to percentage
    return labels, confidences


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the user query from the form
        user_query = request.form.get("query")

        # Predict label and confidence
        labels, confidences = predict_label(model, tokenizer, [user_query])
        predicted_label = labels[0]
        confidence = confidences[0]

        # Render result
        return render_template(
            "index.html", query=user_query, label=predicted_label, confidence=confidence
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)