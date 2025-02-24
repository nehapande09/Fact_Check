from flask import Blueprint, render_template, request, session
from app.utils import zenrow_search, scrape_data, analyze_fact
from model import load_rnn_model_and_tokenizer as load_model_and_tokenizer, predict_label

main = Blueprint('main', __name__)

# Load the RNN model and tokenizer
model, tokenizer = load_model_and_tokenizer()

@main.route("/", methods=["GET", "POST"])
def handle_index():
    error = None
    query = None
    label = None
    confidence = None
    truth_percentage = None
    false_percentage = None
    uncertain_percentage = None
    analysis_result = None
    generative_summary = None
    urls = []

    if request.method == "POST":
        user_query = request.form.get("query")

        if not user_query:
            error = "Please enter a claim."
            return render_template("index.html", error=error)

        # ✅ Use ZenRow API to fetch trending sources
        urls = zenrow_search(user_query)

        # ✅ Store URLs in session for "View Sources"
        session['urls'] = urls if urls else []

        if not urls:
            error = "No sources found for verification."
            return render_template("index.html", error=error)

        # ✅ RNN-Based Fact Prediction (Trending Data)
        try:
            label, confidence = predict_label(model, tokenizer, user_query)
            truth_percentage = round(confidence, 2)
            false_percentage = round(100 - truth_percentage, 2)
        except Exception as e:
            error = f"Error in RNN model prediction: {str(e)}"
            return render_template("index.html", error=error)

        # ✅ ML-Based Analysis (Wikipedia, TOI, BBC)
        try:
            analysis_result = analyze_fact(user_query)
            
            uncertain_percentage = 100 - (truth_percentage + false_percentage) if analysis_result.get("label") == "Uncertain" else 0
            print(analysis_result.get("label"), truth_percentage, false_percentage, uncertain_percentage)
        except Exception as e:
            error = f"Error in ML analysis: {str(e)}"
            return render_template("index.html", error=error)

        # ✅ Generate Final Verdict
        generative_summary = f"Final Verdict: The claim is classified as **{label}** with {confidence}% confidence. ML analysis indicates {analysis_result.get('label')}."

        return render_template(
            "index.html",
            query=user_query,
            label=label,
            confidence=confidence,
            truth_percentage=truth_percentage,
            false_percentage=false_percentage,
            uncertain_percentage=uncertain_percentage,
            urls=urls,
            error=error,
            analysis_result=analysis_result,
            generative_summary=generative_summary
        )

    return render_template("index.html", error=error)
