# Flipkart Review Sentiment Predictor 📊

An interactive web application that predicts the sentiment of Flipkart product reviews. Users can analyze **single reviews** or **batch reviews from CSV files** and get confidence scores for each sentiment category.

---

## Features

- **Single Review Prediction**: Quickly check sentiment for a single Flipkart review.  
- **Batch Prediction**: Upload CSV files with multiple reviews for bulk sentiment analysis.  
- **Confidence Scores**: Displays probability for each sentiment class.  
- **Color-coded Sentiments**: Positive (green), Negative (red), Neutral (gray).  
- **Downloadable Results**: Export predictions as a CSV for further analysis.  
- **User-friendly UI**: Built with Streamlit for seamless interaction.

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
git clone https://github.com/MOHAN1665/Transformer_AI.git
cd Transformer_AI/Amazon  # or the folder containing this app
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage
Run the Streamlit app:
```bash
streamlit run app.py
```
- Single Review: Enter a review and click Predict Sentiment.
- Batch Prediction: Upload a CSV with a Review column, select number of reviews, and predict.
- Download: Export results to CSV after batch predictions.

## How It Works

- Input Handling: Accepts single text input or CSV with multiple reviews.
- Model Prediction: Uses a fine-tuned Transformer model for classification.
- Confidence Calculation: Outputs probabilities for negative, neutral, and positive labels.
- Display:
  - Top predicted sentiment prominently.
  - Full label confidence table for transparency.
  - Color-coded display for quick interpretation.

## Dependencies

- Python 3.8+
- Transformers
- Torch
- Streamlit
- Pandas

## License
This project is open-source under the MIT License.
