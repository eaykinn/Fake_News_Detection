#https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/data

from flask import Flask, render_template, request, jsonify
from backend.importdata import runAlgorithms, GetPerformances
from flask import Flask, request, render_template
import json
import pandas as pd

data = pd.DataFrame


app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/import-data", methods=["POST"])
def import_data():
    haberMetni = request.form.get('habermetni')
    algoritmalar = request.form.get('algoritmalar')
    if not haberMetni:
        return jsonify({"error": "Haber metni boş!"}), 400

    if algoritmalar:
        algoritmalar = json.loads(algoritmalar)
    else:
        algoritmalar = []

    try:
        s1, s2, s3, s4 = runAlgorithms(haberMetni, algoritmalar)
       
        return jsonify({"sonuc1": s1, "sonuc2": s2, "sonuc3": s3, "sonuc4": s4})
    except Exception as e:
        print(f"Hata: {e}")
        return jsonify({"error": "Bir hata oluştu!"}), 500


@app.route("/get-performances", methods=["POST"])
def get_performances():
    algoritmalar = request.form.get('algoritmalar')
    if algoritmalar:
        algoritmalar = json.loads(algoritmalar)
    else:
        algoritmalar = []
    performances = GetPerformances(['Naive Bayes', 'Logistic Regression', 'Decision Tree','KNN'])
    return jsonify(performances)


@app.route("/clear-models", methods=["POST"])
def clear_models():
    import os
    import json
    algoritmalar = request.form.get('algoritmalar')
    if algoritmalar:
        algoritmalar = json.loads(algoritmalar)
    else:
        algoritmalar = []

    model_files = {
        "Naive Bayes": "naive_bayes_model.pkl",
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decisionTree.pkl",
        "KNN": "knn_model.pkl"
    }
    base_dir = os.path.join(os.path.dirname(__file__), "backend", "models")
    cleared = []
    for algo in algoritmalar:
        filename = model_files.get(algo)
        if filename:
            path = os.path.join(base_dir, filename)
            if os.path.exists(path):
             
                with open(path, "wb") as f:
                    pass
                cleared.append(algo)
    return jsonify({"cleared": cleared})


if __name__ == "__main__":
    app.run()