from flask import Flask, request, jsonify, send_file, render_template, make_response
import re
from io import BytesIO
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64
from wordcloud import WordCloud

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)

@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")

@app.route("/predict", methods=["POST"])
def predict():
    predictor = pickle.load(open(r"Models/model_xgb.pkl", "rb"))
    scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))
    cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))
    try:
        if "file" in request.files:
            file = request.files["file"]
            data = pd.read_csv(file)

            predictions, graphs = bulk_prediction(predictor, scaler, cv, data)

            response = make_response(predictions.getvalue())
            response.headers["Content-Disposition"] = "attachment; filename=Predictions.csv"
            response.headers["Content-Type"] = "text/csv"
            response.headers["X-Graph-Exists"] = "true"

            graph_data = {f"X-Graph-Data-{i}": base64.b64encode(graph.getvalue()).decode("ascii") for i, graph in enumerate(graphs)}
            for key, value in graph_data.items():
                response.headers[key] = value

            return response

        elif "text" in request.json:
            text_input = request.json["text"]
            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)
            return jsonify({"prediction": predicted_sentiment})

    except Exception as e:
        return jsonify({"error": str(e)})

def single_prediction(predictor, scaler, cv, text_input):
    corpus = []
    stemmer = PorterStemmer()
    review = re.sub("[^a-zA-Z]", " ", text_input)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
    review = " ".join(review)
    corpus.append(review)
    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)[0]

    return "Positive" if y_predictions == 1 else "Negative"

def bulk_prediction(predictor, scaler, cv, data):
    corpus = []
    stemmer = PorterStemmer()
    for i in range(0, data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["Sentence"])
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)
    y_predictions = predictor.predict_proba(X_prediction_scl)
    y_predictions = y_predictions.argmax(axis=1)
    y_predictions = list(map(sentiment_mapping, y_predictions))

    data["Predicted sentiment"] = y_predictions
    predictions_csv = BytesIO()
    data.to_csv(predictions_csv, index=False)
    predictions_csv.seek(0)

    sentiment_distribution_graph = get_distribution_graph(data)
    sentiment_bar_chart = get_bar_chart(data)
    word_cloud = get_word_cloud(corpus)

    return predictions_csv, [sentiment_distribution_graph, sentiment_bar_chart, word_cloud]

def get_distribution_graph(data):
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.01, 0.01)

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
    )

    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()

    graph.seek(0)
    return graph

def get_bar_chart(data):
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    tags = data["Predicted sentiment"].value_counts()

    tags.plot(
        kind="bar",
        color=colors,
        xlabel="Sentiment",
        ylabel="Count",
    )

    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()

    graph.seek(0)
    return graph

def get_word_cloud(corpus):
    text = " ".join(corpus)
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", max_words=100).generate(text)

    fig = plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()

    graph.seek(0)
    return graph

def sentiment_mapping(x):
    return "Positive" if x == 1 else "Negative"

if __name__ == "__main__":
    app.run(port=5000, debug=True)
