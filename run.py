from flask import Flask
from app.routes import handle_index

app = Flask(__name__)
app.secret_key = 'your_secret_key'

@app.route("/", methods=["GET", "POST"])
def index():
    return handle_index()

if __name__ == "__main__":
    app.run(debug=True)
