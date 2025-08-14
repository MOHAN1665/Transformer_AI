import streamlit as st
from transformers import pipeline
import torch
import pandas as pd

# Define label classes (ensure this matches your model training)
label_classes = ['negative', 'neutral', 'positive']
label_map = {f"LABEL_{i}": label for i, label in enumerate(label_classes)}

# Load pipeline once and cache it
@st.cache_resource
def load_classifier():
    device_id = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text-classification",
        model="fine-tuned-amazon-model",
        tokenizer="fine-tuned-amazon-model",
        device=device_id,
        top_k=None  # get all label scores
    )

classifier = load_classifier()

# Streamlit UI
st.set_page_config(page_title="Amazon Review Sentiment Classifier 🛒", layout="centered")
st.title("🛒 Amazon Review Sentiment Classifier")
st.markdown(
    """
    Enter an Amazon product review below and get sentiment predictions along with confidence scores.
    """
)

st.sidebar.title("About")
st.sidebar.info(
    """
    This app uses a DistilBERT-based model fine-tuned on Amazon reviews for sentiment classification.

    - Labels: Negative, Neutral, Positive  
    """
)

user_input = st.text_area("Enter your review text here:", height=150)

if st.button("🧠 Predict Sentiment"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        try:
            # Get predictions for all classes
            results = classifier(user_input)[0]

            # Map labels and build DataFrame
            data = []
            for res in results:
                label_name = label_map.get(res['label'], res['label'])
                score = res['score']
                data.append((label_name, score))

            df = pd.DataFrame(data, columns=["Sentiment", "Confidence"])
            df = df.sort_values(by="Confidence", ascending=False).reset_index(drop=True)

            # Display top sentiment prominently
            top = df.iloc[0]
            st.success(f"**Top Sentiment:** `{top.Sentiment}` with confidence `{top.Confidence:.4f}`")

            # Show full confidence table
            st.markdown("### All Sentiment Scores")
            st.table(df.style.format({"Confidence": "{:.4f}"}))

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
