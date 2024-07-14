import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import base64

# flask --app api.py run --port=5000
prediction_endpoint = "http://127.0.0.1:5000/predict"

st.title("Text Sentiment Predictor")

uploaded_file = st.file_uploader(
    "Choose a CSV file for bulk prediction - Upload the file and click on Predict",
    type="csv",
)

# Text input for sentiment prediction
user_input = st.text_input("Enter text and click on Predict", "")

# Function to display graphs
def display_graphs(response):
    if 'X-Graph-Data-0' in response.headers:
        graph_data1 = response.headers['X-Graph-Data-0']
        graph_data2 = response.headers['X-Graph-Data-1']
        graph_data3 = response.headers['X-Graph-Data-2']

        graph_urls = [
            "data:image/png;base64," + graph_data1,
            "data:image/png;base64," + graph_data2,
            "data:image/png;base64," + graph_data3,
        ]

        for graph_url in graph_urls:
            graph_image = Image.open(BytesIO(base64.b64decode(graph_url.split(",")[1])))
            st.image(graph_image, use_column_width=True)

# Prediction on single sentence
if st.button("Predict"):
    if uploaded_file is not None:
        file = {"file": uploaded_file}
        response = requests.post(prediction_endpoint, files=file)
        response_bytes = BytesIO(response.content)
        response_df = pd.read_csv(response_bytes)

        st.download_button(
            label="Download Predictions",
            data=response_bytes,
            file_name="Predictions.csv",
            key="result_download_button",
        )
        display_graphs(response)

    else:
        response = requests.post(prediction_endpoint, data={"text": user_input})
        response = response.json()
        st.write(f"Predicted sentiment: {response['prediction']}")