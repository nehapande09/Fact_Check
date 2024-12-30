from app.utils import google_search, fetch_news_articles, correct_grammar, generate_converse, predict_label, \
    extract_text_from_image, fetch_historical_facts, fetch_current_data
from model import load_model_and_tokenizer
from flask import Blueprint, render_template, request, session


main = Blueprint('main', __name__)

model, tokenizer = load_model_and_tokenizer()

model, tokenizer = load_model_and_tokenizer()

def handle_index():
    truth_percentage = None
    false_percentage = None
    error = None
    urls = []
    query = None
    corrected_query = None
    label = None
    confidence = None
    converse_query = None
    converse_label = None
    converse_confidence = None

    if request.method == "POST":
        input_type = request.form.get("input_type")  # 'news' or 'other'
        image = request.files.get("image")
        user_query = request.form.get("query")

        if image and not user_query:
            try:
                user_query = extract_text_from_image(image)
                print("Extracted text:", user_query)
            except Exception as e:
                print(f"Error extracting text from image: {e}")
                error = "Failed to process the uploaded image."
                return render_template("index.html", error=error)

        if not user_query:
            error = "Please enter a claim or upload an image."
            return render_template("index.html", error=error)

        corrected_query = correct_grammar(user_query)
        converse_query = generate_converse(corrected_query)

        if input_type == "news":
            urls = fetch_news_articles(corrected_query, num_results=5)
            if not urls:  # Fallback to Google search
                urls = google_search(corrected_query, num_results=5)
        else:
            urls = google_search(corrected_query, num_results=5)

        if input_type == "current":
            urls = fetch_current_data(corrected_query, num_results=5)
            if not urls:  # Fallback to Google search
                urls = google_search(corrected_query, num_results=5)
        else:
            urls = google_search(corrected_query, num_results=5)



        if not urls:
            error = "No search results found."
            return render_template("    index.html", error=error)

        session['urls'] = urls

        label, confidence = predict_label(model, tokenizer, corrected_query)
        converse_label, converse_confidence = predict_label(model, tokenizer, converse_query)

        truth_percentage = round((confidence + (100 - converse_confidence)) / 2, 2)
        false_percentage = round(100 - truth_percentage, 2)

        is_true = truth_percentage > 50

        return render_template(
            "index.html",
            query=user_query,
            corrected_query=corrected_query,
            label=label,
            confidence=confidence,
            converse_query=converse_query,
            converse_label=converse_label,
            converse_confidence=converse_confidence,
            truth_percentage=truth_percentage,
            false_percentage=false_percentage,
            is_true=is_true,
            urls=urls,
            error=error
        )

    return render_template(
        "index.html",
        truth_percentage=truth_percentage,
        false_percentage=false_percentage,
        error=error
    )