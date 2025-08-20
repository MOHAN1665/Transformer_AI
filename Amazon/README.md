# Amazon Review Sentiment Classifier 🛒

A web application that analyzes Amazon product reviews and classifies them into sentiment categories using a **fine-tuned Transformer model**. The app provides sentiment predictions along with confidence scores in an interactive interface.

---

## Features

- **Sentiment Analysis**: Classifies reviews as **Negative**, **Neutral**, or **Positive**.  
- **Confidence Scores**: Displays model confidence for each sentiment label.  
- **Top Prediction Highlight**: Clearly shows the most probable sentiment.  
- **Interactive UI**: Built with **Streamlit** for easy user interaction.  
- **Fast & Lightweight**: Uses a fine-tuned **DistilBERT** model for quick predictions.

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
git clone https://github.com/MOHAN1665/Transformer_AI/tree/main/Amazon
cd Amazon
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
- Enter an Amazon product review in the text area.
- Click Predict Sentiment to see results.
- The app displays:
  - Top sentiment prediction with confidence.
  - Full table of all sentiment scores.
 
## How It Works

- Text Input: User enters a product review.
- Model Prediction: Fine-tuned DistilBERT predicts sentiment probabilities for all classes.
- Display Results: Top sentiment highlighted and full confidence table displayed for transparency.

## Dependencies

- Python 3.8+
- Transformers
- Streamlit
- Torch
- Pandas

## License
This project is open-source and available under the MIT License.
