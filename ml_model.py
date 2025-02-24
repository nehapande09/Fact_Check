import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

# ✅ Load sample dataset (Replace this with your own dataset)
data = fetch_20newsgroups(subset='train', categories=['sci.space', 'rec.autos'], remove=('headers', 'footers', 'quotes'))
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# ✅ Train TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

# ✅ Train ML Model (Logistic Regression as an example)
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# ✅ Save Model & Vectorizer
with open("ml_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("✅ Model and vectorizer saved successfully!")
