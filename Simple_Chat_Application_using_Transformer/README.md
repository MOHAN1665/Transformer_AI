# Mini Transformer Chat

A lightweight, memory-augmented chatbot built using Python and Streamlit. This project demonstrates a **transformer-inspired question encoder** and a simple **focus-based decoder** for answering user queries with a dynamically updateable memory.

---

## Features

- **Question Encoding**: Tokenizes and normalizes user input, calculates attention scores between words.
- **Focus Map**: Highlights which words are most influential in determining the answer.
- **Memory-Based Answers**: Retrieves answers from a JSON-based database (`qa_memory.json`) with best-match logic.
- **Dynamic Memory Update**: Add new questions and answers directly through the interface.
- **Lightweight**: Minimal dependencies, runs locally without large pretrained models.

---

## Table of Contents

1. [Installation](#installation)  
2. [Usage](#usage)  
3. [How It Works](#how-it-works)  
4. [Memory Update](#memory-update)  
5. [Dependencies](#dependencies)  
6. [License](#license)  

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/MOHAN1665/Transformer_AI.git
cd Transformer_AI/Aadhaar_Services_Chatbot
```

2. Create a virtual environment (recommended):

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
- Enter your query in the input box.
- View Tokens and Focus Map in the encoder section.
- Get the best matched answer from memory.
- Add new questions and answers to memory directly via the interface.

## How It Works

1. Normalization & Tokenization:
   Converts input to lowercase, removes punctuation, splits into tokens.

2. Attention Score Calculation:
   Each token is compared to others using predefined embeddings to determine word importance.

3. Best Match Retrieval:
   Compares token sets with existing database entries to find the closest matching question.

4. Decoder:
   Fetches the answer associated with the best matching question from memory.
