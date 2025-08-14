import streamlit as st
from transformers import pipeline
import joblib
import pandas as pd

# Load label encoder (must be saved during training)
@st.cache_resource
def load_label_encoder(path="label_encoder.pkl"):
    return joblib.load(path)

@st.cache_resource
def load_pipeline(model_dir="fine-tuned-airline-model"):
    return pipeline("text-classification", model=model_dir, tokenizer=model_dir, top_k=None)

label_encoder = load_label_encoder()
label_map = {f"LABEL_{i}": name for i, name in enumerate(label_encoder.classes_)}
clf = load_pipeline()

# Page config and sidebar info
st.set_page_config(page_title="Airline Tweet Sentiment Classifier ✈️", layout="centered")

st.sidebar.title("About")
st.sidebar.info(
    """
    This app classifies airline tweets into sentiment categories using a fine-tuned Transformer model.

    - Model: DistilBERT fine-tuned on airline tweets  
    - Labels: Positive, Neutral, Negative  

    """
)

dark_mode = st.sidebar.checkbox("Enable Dark Mode", value=False)
if dark_mode:
    st.markdown(
        """
        <style>
        .reportview-container {
            background-color: #0e1117;
            color: #fafafa;
        }
        .stButton>button {
            background-color: #007ACC;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True
    )

st.title("✈️ Airline Tweet Sentiment Classifier")
st.markdown(
    "Paste or type an airline-related tweet below and get sentiment predictions with confidence scores."
)

tweet = st.text_area("✍️ Your Tweet:", height=120)

if st.button("🧠 Predict Sentiment"):
    if not tweet.strip():
        st.warning("⚠️ Please enter a tweet to analyze.")
    else:
        with st.spinner("Analyzing tweet..."):
            try:
                results = clf(tweet)[0]  # pipeline returns list of dicts per sample

                # Build dataframe with labels and scores
                data = []
                for res in results:
                    label_name = label_map.get(res['label'], res['label'])
                    score = res['score']
                    data.append((label_name, score))

                df = pd.DataFrame(data, columns=["Sentiment", "Confidence"])
                df = df.sort_values(by="Confidence", ascending=False).reset_index(drop=True)

                # Show top prediction prominently
                top_pred = df.iloc[0]
                st.success(f"**Top Prediction:** `{top_pred.Sentiment}` with confidence `{top_pred.Confidence:.2f}`")

                # Show full sorted table of all predictions
                st.markdown("### Full Sentiment Scores")
                st.table(df.style.format({"Confidence": "{:.4f}"}))

            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")

# Footer or optional additional info
st.markdown("---")
st.markdown("© 2025 Task / Mohan — Built with Streamlit and 🤗 Transformers")
