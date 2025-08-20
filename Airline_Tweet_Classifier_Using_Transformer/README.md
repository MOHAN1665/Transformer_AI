# Airline Tweet Sentiment Classifier ✈️

A web application that classifies airline-related tweets into sentiment categories using a **fine-tuned Transformer model**. The app provides sentiment predictions along with confidence scores in an interactive interface.

---

## Features

- **Sentiment Analysis**: Classifies tweets into **Positive**, **Neutral**, or **Negative** sentiment.  
- **Confidence Scores**: Shows confidence for each sentiment label.  
- **Top Prediction Highlight**: Highlights the most probable sentiment prominently.  
- **Interactive UI**: Built with **Streamlit** for easy user interaction.  
- **Dark Mode Support**: Optional dark mode for comfortable viewing.  
- **Lightweight & Fast**: Uses a fine-tuned **DistilBERT** model for quick predictions.

---

## Table of Contents

1. [Installation](#installation)  
2. [Usage](#usage)  
3. [How It Works](#how-it-works)  
4. [Dependencies](#dependencies)  
5. [License](#license)  

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/MOHAN1665/Transformer_AI/tree/main/Airline_Tweet_Classifier_Using_Transformer
cd Airline_Tweet_Classifier_Using_Transformer
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```
- Paste or type an airline tweet in the input box.
- Click Predict Sentiment to see results.
- The app displays:
  - Top sentiment prediction with confidence.
  - Full table with all label confidences.
- Optional: Enable Dark Mode from the sidebar for better readability.

## How It Works

- Text Input: User enters an airline-related tweet.
- Label Encoding: Pre-trained label encoder maps model output labels to human-readable sentiments.
- Model Prediction: Fine-tuned DistilBERT classifies the tweet and outputs probabilities for each sentiment.
- Display Results: Top sentiment highlighted and all predictions shown in a sorted table.

## Dependencies

- Python 3.8+
- Transformers
- Streamlit
- Joblib
- Pandas

## License
This project is open-source and available under the MIT License.
