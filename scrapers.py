import requests
from bs4 import BeautifulSoup
import re
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


# Function to search the web for the input query
def search_web(query):
    search_url = f"https://www.google.com/search?q={query}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract links from the search results
    links = []
    for a_tag in soup.select('a'):
        href = a_tag.get('href', '')
        match = re.search(r'/url\?q=(.*?)&', href)
        if match:
            url = match.group(1)
            if "http" in url:
                links.append(url)
        if len(links) >= 5:  # Limit to first 5 websites
            break
    return links


# Load the trained RNN model and tokenizer
def load_model_and_tokenizer():
    model = load_model("rnn_model.h5")
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
    return model, tokenizer


# Predict labels for claims
def predict_label(model, tokenizer, claims):
    sequences = tokenizer.texts_to_sequences(claims)
    max_length = 100  # Ensure consistency with training
    data = pad_sequences(sequences, maxlen=max_length)
    predictions = model.predict(data)
    labels = ["SUPPORTS" if pred > 0.5 else "REFUTES" for pred in predictions]
    return labels, predictions


# Main workflow
def main():
    # Get user input
    input_query = input("Enter your query: ")

    # Step 1: Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    # Step 2: Predict the label for the claim
    print("Predicting label for the claim...")
    labels, predictions = predict_label(model, tokenizer, [input_query])

    # Extract the claim prediction
    claim_prediction = labels[0]
    claim_probability = predictions[0]

    # Step 3: Display the results
    print("\nResults:")
    print(f"Claim: {input_query}")
    print(f"Predicted Label: {claim_prediction}")
    print(f"Confidence: {claim_probability[0] * 100:.2f}%")


if __name__ == "__main__":
    main()
