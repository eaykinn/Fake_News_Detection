import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
import re
from pympler import asizeof
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer
import joblib
import time

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_model(filename):
    try:
        data = joblib.load(filename)
        return data['model'], data['vectorizer'], data['metrics']
    except EOFError:
        print(f"Model missing...: {filename}")
        raise

def save_model(model, vectorizer, filename, metrics=None):
    data = {'model': model, 'vectorizer': vectorizer}
    if metrics is not None:
        data['metrics'] = metrics
    joblib.dump(data, filename)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def importData(algoritmalar):
    model_files = {
        "Naive Bayes": "naive_bayes_model.pkl",
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decisionTree.pkl",
        "KNN": "knn_model.pkl"
    }
    eksik_modeller = []
    mevcut_modeller = {}
    for algo in algoritmalar:
        try:
            model_path = os.path.join(BASE_DIR, 'backend', 'models', model_files[algo])
            modelx, vec, met = load_model(model_path)
            mevcut_modeller[algo] = (modelx, vec, met)
        except Exception:
            eksik_modeller.append(algo)


    if eksik_modeller:
        truepath = os.path.join(BASE_DIR, 'data', 'True.csv')
        trueNewsDataSet = pd.read_csv(truepath)
        trueNewsDataSet['label'] = 1
        falsepath = os.path.join(BASE_DIR, 'data', 'Fake.csv')
        fakeNewsDataSet = pd.read_csv(falsepath)
        fakeNewsDataSet['label'] = 0

        combinedDataSet = pd.concat([trueNewsDataSet, fakeNewsDataSet], ignore_index=True)
        combinedDataSet = combinedDataSet.drop_duplicates().reset_index(drop=True)
        combinedDataSet = combinedDataSet.dropna(subset=['text'])
        combinedDataSet = combinedDataSet[combinedDataSet['text'].str.strip() != ""]
        combinedDataSet = combinedDataSet.sample(frac=1).reset_index(drop=True)
        combinedDataSet['text'] = combinedDataSet['text'].apply(clean_text)

        X_train_text, X_test_text, y_train, y_test = train_test_split(
            combinedDataSet['text'], combinedDataSet['label'], test_size=0.2, random_state=42
        )
        vectorizer = TfidfVectorizer(max_features=5000)
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)
    else:
        X_train = X_test = y_train = y_test = vectorizer = None

    return X_train, X_test, y_train, y_test, vectorizer, eksik_modeller, mevcut_modeller


def runAlgorithms(haberMetni, algoritmalar):
    X_train, X_test, y_train, y_test, vectorizer, eksik_modeller, mevcut_modeller = importData(algoritmalar)
    results = [None, None, None, None]
    metrics = [None, None, None, None]
    algo_list = ["Naive Bayes", "Logistic Regression", "Decision Tree", "KNN"]

    icerik = clean_text(haberMetni)

    for idx, algo in enumerate(algo_list):
        if algo not in algoritmalar:
            continue
        if algo in mevcut_modeller:
           
            model, vec, met = mevcut_modeller[algo]
            X_examp = vec.transform([icerik])
            y_examp = model.predict(X_examp)
            result = True if y_examp == 1 else False
            results[idx] = result
            metrics[idx] = met
        elif algo in eksik_modeller:
            # Model eksik, eğit ve kaydet
            if algo == "Naive Bayes":
                nBresult, nbMet = naiveBayes(X_train, X_test, y_train, y_test, haberMetni, vectorizer)
                results[0] = nBresult
                metrics[0] = nbMet
            elif algo == "Logistic Regression":
                lrResult, lbMet = logisticRegression(X_train, X_test, y_train, y_test, haberMetni, vectorizer)
                results[1] = lrResult
                metrics[1] = lbMet
            elif algo == "Decision Tree":
                dcResult, dtMet = decisionTree(X_train, X_test, y_train, y_test, haberMetni, vectorizer)
                results[2] = dcResult
                metrics[2] = dtMet
            elif algo == "KNN":
                kNNResult, knnMet = kNN(X_train, X_test, y_train, y_test, haberMetni, vectorizer)
                results[3] = kNNResult
                metrics[3] = knnMet

    return tuple(results)

def naiveBayes(X_train, X_test, y_train, y_test, haberMetni, vectorizer):
    start_time = time.time()
    modelNB = MultinomialNB()
    modelNB.fit(X_train, y_train)
    model_memory = asizeof.asizeof(modelNB) / (1024 * 1024)
    y_pred = modelNB.predict(X_test)
    duration = time.time() - start_time
    metric = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "duration_seconds": duration,
        "memory_usage_bytes": model_memory
    }
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted', zero_division=0),
        'recall': make_scorer(recall_score, average='weighted', zero_division=0),
        'f1': make_scorer(f1_score, average='weighted', zero_division=0)
    }
    cv_results = cross_validate(
        MultinomialNB(),
        X_train, y_train, cv=5, scoring=scoring
    )
    metric["cross_val"] = {
        "accuracy": cv_results['test_accuracy'].mean(),
        "precision": cv_results['test_precision'].mean(),
        "recall": cv_results['test_recall'].mean(),
        "f1": cv_results['test_f1'].mean()
    }
    icerik = clean_text(haberMetni)
    X_examp = vectorizer.transform([icerik])
    y_examp = modelNB.predict(X_examp)
    sonuc = True if y_examp == 1 else False
    modelpath = os.path.join(BASE_DIR, 'backend', 'models', 'naive_bayes_model.pkl')
    save_model(modelNB, vectorizer, modelpath, metric)
    return sonuc, metric

def logisticRegression(X_train, X_test, y_train, y_test, haberMetni, vectorizer):
    start_time = time.time()
    modelLR = LogisticRegression()
    modelLR.fit(X_train, y_train)
    model_memory = asizeof.asizeof(modelLR) / (1024 * 1024)
    y_pred = modelLR.predict(X_test)
    duration = time.time() - start_time
    metric = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "duration_seconds": duration,
        "memory_usage_bytes": model_memory
    }
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted', zero_division=0),
        'recall': make_scorer(recall_score, average='weighted', zero_division=0),
        'f1': make_scorer(f1_score, average='weighted', zero_division=0)
    }
    cv_results = cross_validate(
        LogisticRegression(),
        X_train, y_train, cv=5, scoring=scoring
    )
    metric["cross_val"] = {
        "accuracy": cv_results['test_accuracy'].mean(),
        "precision": cv_results['test_precision'].mean(),
        "recall": cv_results['test_recall'].mean(),
        "f1": cv_results['test_f1'].mean()
    }
    icerik = clean_text(haberMetni)
    X_examp = vectorizer.transform([icerik])
    y_examp = modelLR.predict(X_examp)
    modelpath = os.path.join(BASE_DIR, 'backend', 'models', 'logistic_regression.pkl')
    save_model(modelLR, vectorizer, modelpath, metric)
    sonuc = True if y_examp == 1 else False
    return sonuc, metric

def decisionTree(X_train, X_test, y_train, y_test, haberMetni, vectorizer):
    start_time = time.time()
    clf = DecisionTreeClassifier(random_state=42, max_depth=3, min_samples_leaf=2)
    clf.fit(X_train, y_train)
    model_memory = asizeof.asizeof(clf) / (1024 * 1024)
    y_pred = clf.predict(X_test)
    duration = time.time() - start_time
    metric = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "duration_seconds": duration,
        "memory_usage_bytes": model_memory
    }
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted', zero_division=0),
        'recall': make_scorer(recall_score, average='weighted', zero_division=0),
        'f1': make_scorer(f1_score, average='weighted', zero_division=0)
    }
    cv_results = cross_validate(
        DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_leaf=3),
        X_train, y_train, cv=5, scoring=scoring
    )
    metric["cross_val"] = {
        "accuracy": cv_results['test_accuracy'].mean(),
        "precision": cv_results['test_precision'].mean(),
        "recall": cv_results['test_recall'].mean(),
        "f1": cv_results['test_f1'].mean()
    }
    icerik = clean_text(haberMetni)
    X_examp = vectorizer.transform([icerik])
    y_examp = clf.predict(X_examp)
    sonuc = True if y_examp == 1 else False
    modelpath = os.path.join(BASE_DIR, 'backend', 'models', 'decisionTree.pkl')
    save_model(clf, vectorizer, modelpath, metric)
    return sonuc, metric

def kNN(X_train, X_test, y_train, y_test, haberMetni, vectorizer):
    start_time = time.time()
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    model_memory = asizeof.asizeof(knn) / (1024 * 1024)
    y_pred = knn.predict(X_test)
    duration = time.time() - start_time
    metric = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "duration_seconds": duration,
        "memory_usage_bytes": model_memory
    }
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='weighted', zero_division=0),
        'recall': make_scorer(recall_score, average='weighted', zero_division=0),
        'f1': make_scorer(f1_score, average='weighted', zero_division=0)
    }
    cv_results = cross_validate(
        KNeighborsClassifier(n_neighbors=5),
        X_train, y_train, cv=5, scoring=scoring
    )
    metric["cross_val"] = {
        "accuracy": cv_results['test_accuracy'].mean(),
        "precision": cv_results['test_precision'].mean(),
        "recall": cv_results['test_recall'].mean(),
        "f1": cv_results['test_f1'].mean()
    }
    icerik = clean_text(haberMetni)
    X_examp = vectorizer.transform([icerik])
    y_examp = knn.predict(X_examp)
    modelpath = os.path.join(BASE_DIR, 'backend', 'models', 'knn_model.pkl')
    save_model(knn, vectorizer, modelpath, metric)
    sonuc = True if y_examp == 1 else False
    return sonuc, metric

def GetPerformances(algoritmalar):
    performances = {}
    model_files = {
        "Naive Bayes": "naive_bayes_model.pkl",
        "Logistic Regression": "logistic_regression.pkl",
        "Decision Tree": "decisionTree.pkl",
        "KNN": "knn_model.pkl"
    }
    for algo in algoritmalar:
        modelpath = os.path.join(BASE_DIR, 'backend', 'models', model_files[algo])
        try:
            model, vec, met = load_model(modelpath)
            performances[algo] = {
                "accuracy": met.get("accuracy"),
                "duration": met.get("duration_seconds"),
                "classification_report": met.get("classification_report"),
                "cross_val": met.get("cross_val"),
                "confusion_matrix": met.get("confusion_matrix"),
                "memory_usage_bytes": met.get("memory_usage_bytes")
            }
        except EOFError:
            print(f"Model missing...: {modelpath}")
    return performances