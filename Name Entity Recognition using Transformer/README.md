# Named Entity Recognition (NER) Application

A web-based NER tool built using **Gradio** and a fine-tuned **Transformer model**. The application extracts entities from legal, financial, or general text, providing accurate results in an easy-to-use interface.

---

## Features

- **Entity Extraction**: Detects entities such as `PERSON`, `ORG`, `LOCATION`, `DATE`, `TIME`, `MONEY`, `CLAUSE`, `SECTION`, and more.
- **Custom Regex Support**: Captures patterns not always detected by the model, including dates, times, and financial amounts.
- **Interactive Web UI**: View extracted entities in a neatly formatted table with entity type and confidence score.
- **Accurate and Reliable**: Leveraging our fine-tuned Transformer model for precise entity recognition across multiple domains.
- **Easy to Use**: Paste text, click analyze, and get instant results.

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
cd Transformer_AI/Name_Entity_Recognition_using_Transformer
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
Run the Gradio app

## How It Works

1. Text Processing:
  Input text is cleaned and tokenized for the model.

2. NER Model:
  Uses a fine-tuned Transformer model to predict entities with high accuracy.

3. Custom Pattern Matching:
  Regex patterns detect additional entities like dates, times, clauses, and financial figures that may not be explicitly labeled by the model.

4. Result Display:
  Outputs a table with Entity, Type, and Score, sorted by appearance in the text.

## Dependencies

- Python 3.8+
- Transformers
- Gradio
- regex for pattern matching

## License
This project is open-source and available under the MIT License.
