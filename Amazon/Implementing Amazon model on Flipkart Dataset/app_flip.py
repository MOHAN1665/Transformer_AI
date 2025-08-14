import streamlit as st
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from io import StringIO

# Load model/tokenizer once
@st.cache_resource(show_spinner=False)
def load_model_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

def predict_batch(texts, tokenizer, model, device):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
    confidences, predictions = torch.max(probs, dim=1)
    return predictions.cpu().tolist(), confidences.cpu().tolist()

def predict_single(text, tokenizer, model, device, label_map):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]
    scores = {label_map[i]: float(probs[i]) for i in range(len(probs))}
    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    top_label = next(iter(sorted_scores))
    top_score = sorted_scores[top_label]
    return top_label, top_score, sorted_scores

def sentiment_color(sentiment):
    if sentiment == "positive":
        return "green"
    elif sentiment == "negative":
        return "red"
    else:
        return "gray"

def main():
    st.set_page_config(page_title="📊 Flipkart Sentiment Analysis", layout="wide")
    
    st.markdown("<h1 style='text-align:center; color:#4B8BBE;'>📊 Flipkart Review Sentiment Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:18px;'>Upload your CSV file or enter a review to predict sentiments.</p>", unsafe_allow_html=True)
    st.markdown("---")

    model_dir = "fine-tuned-amazon-model"
    tokenizer, model = load_model_tokenizer(model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

    # --- Upload CSV and batch predict ---
    with st.container():
        st.subheader("🗂️ Batch Prediction from CSV")
        uploaded_file = st.file_uploader("Upload CSV file (must contain 'Review' column)", type=['csv'])
        if uploaded_file:
            df = pd.read_csv(uploaded_file, encoding='latin1')
            if 'Review' not in df.columns:
                st.error("❌ CSV file must contain a 'Review' column.")
            else:
                st.success(f"✅ Loaded {len(df)} reviews.")
                max_rows = st.slider("Select number of reviews to predict:", 1, len(df), 10)
                
                if st.button("Predict Batch Sentiments"):
                    texts = df['Review'].head(max_rows).tolist()
                    with st.spinner(f"Predicting {max_rows} reviews..."):
                        preds, confs = predict_batch(texts, tokenizer, model, device)
                    pred_labels = [label_map.get(p, str(p)) for p in preds]
                    
                    results_df = pd.DataFrame({
                        'Review': texts,
                        'Predicted Sentiment': pred_labels,
                        'Confidence': [f"{c:.3f}" for c in confs]
                    })

                    # Colorful display
                    def color_sentiment(val):
                        color = sentiment_color(val)
                        return f'color: {color}; font-weight: bold;'

                    st.dataframe(results_df.style.applymap(color_sentiment, subset=['Predicted Sentiment']), height=400)
                    
                    # Download CSV
                    csv_buffer = StringIO()
                    results_df.to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv_buffer.getvalue(),
                        file_name="flipkart_sentiment_predictions.csv",
                        mime="text/csv"
                    )

    st.markdown("---")

    # --- Single text prediction ---
    with st.container():
        st.subheader("✍️ Single Review Sentiment Prediction")
        user_text = st.text_area("Enter your review text here:", height=120)
        if st.button("Predict Sentiment for Single Review"):
            if not user_text.strip():
                st.warning("⚠️ Please enter some text!")
            else:
                with st.spinner("Analyzing your review..."):
                    top_label, top_score, all_scores = predict_single(user_text, tokenizer, model, device, label_map)
                st.markdown(f"### Top Predicted Sentiment: <span style='color:{sentiment_color(top_label)}'>{top_label.capitalize()}</span> (Confidence: {top_score:.3f})", unsafe_allow_html=True)
                st.markdown("**All Label Scores:**")
                for label, score in all_scores.items():
                    color = sentiment_color(label)
                    st.markdown(f"- <span style='color:{color}'>{label.capitalize()}</span>: {score:.3f}", unsafe_allow_html=True)

    st.markdown("<br><br><hr><p style='text-align:center; color:#888;'>Made by Mohan</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
