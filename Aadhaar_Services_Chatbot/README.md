# Aadhaar Services Assistant Chatbot 🤖

A conversational AI assistant designed to provide information and support for **Aadhaar-related services in India**. This chatbot leverages advanced natural language processing to answer queries about Aadhaar updates, e-KYC services, mAadhaar app, PVC cards, and more.
![Aadhaar Services Assistant Chatbot](home.png)

---

## Features

- **Natural Language Understanding**: Handles user queries about Aadhaar services.  
- **Contextual Conversations**: Maintains conversation context for follow-up questions.  
- **FAQ Retrieval**: Provides relevant answers from a comprehensive Aadhaar knowledge base.  
- **Multi-modal Interaction**: Supports both text and voice input/output.  
- **Security Awareness**: Detects and warns about sensitive information requests.  
- **User-Friendly Interface**: Clean, intuitive **Gradio-based** web interface.  

---

## Technical Architecture

### Core Components

- **T5 Transformer Model**: Fine-tuned for Aadhaar-specific question answering.  
- **Sentence Transformers**: For semantic similarity search.  
- **FAISS**: Efficient vector similarity search for FAQ retrieval.  
- **Conversation Memory**: Tracks context across dialogue turns.  
- **Speech Processing**: Optional voice input/output using Whisper and pyttsx3.  

### Key Technologies

- Python 3.8+  
- PyTorch  
- Transformers (Hugging Face)  
- Gradio  
- FAISS  
- SentenceTransformers  

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/MOHAN1665/Transformer_AI/tree/main/Aadhaar_Services_Chatbot
cd Aadhaar_Services_Chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Key Dependencies:

- torch
- transformers
- sentence-transformers
- faiss-cpu (or faiss-gpu for CUDA)
- gradio
- pandas
- openai-whisper (optional, for speech-to-text)
- pyttsx3 (optional, for text-to-speech)

## Data Preparation

- Prepare your FAQ dataset as a CSV with columns:
  - question: The user question
  -  answer: Corresponding answer
  - official_link: Optional official reference link
- Place the CSV file in the project directory (default: aadhaar_dataset_final1_upd.csv).

## Model Setup

- Place your fine-tuned T5 model in the aadhaar_t5_lora_final directory.
- The model folder should include:
  - PyTorch model weights
  - Tokenizer files
  - Configuration files
 
## Usage
Running the Application
```bash
python app.py
```
` The app will start a local server (default: http://127.0.0.1:7860) with a web interface.

## Interacting with the Chatbot
- Text Input: Type your question and press Enter or click Send.
- Voice Input: Click the microphone button to record your question.
- Voice Output: Enable "Speak replies" to hear the assistant’s responses.
- Suggested Questions: Click on any predefined question to quickly ask common queries.

## Example Questions:
- "How to update my address in Aadhaar?"
- "What documents are needed for mobile number update?"
- "How to download e-Aadhaar?"
- "What is the fee for PVC card?"
- "How to lock biometric data?"

## Project Structure
```bash
aadhaar-chatbot/
├── app.py                         # Main application file
├── aadhaar_t5_lora_final/         # Fine-tuned T5 model directory
├── aadhaar_dataset_final1_upd.csv # FAQ dataset
├── requirements.txt               # Python dependencies
├── tts_cache/                     # Generated audio files (runtime)
└── README.md                      # This file
```

## Customization

- Modifying FAQ Database: Edit the CSV to add new questions/answers.
- Adjusting Conversation Memory: Modify max_turns in ConversationMemory.
- Changing the Model: Replace files in aadhaar_t5_lora_final with your custom fine-tuned model.

## Security Features

- Detects requests for sensitive information (OTP, full Aadhaar numbers, passwords).
- Provides security warnings when sensitive topics are detected.
- Refers users to official channels for sensitive operations.

## Performance Considerations

- GPU-enabled environment recommended for faster responses.
- FAQ indexing occurs at startup; large datasets may increase load time.
- Voice processing requires additional dependencies and may affect performance.

## Limitations

- Accuracy depends on the quality of the fine-tuned model and FAQ database.
- Voice recognition may vary with accent and audio quality.
- Complex multi-part questions may not be handled perfectly.

## Support

- Model performance: Ensure proper fine-tuning and dataset preparation.
- Installation issues: Verify dependency compatibility.
- Functionality: Review conversation memory and FAQ retrieval logic.

## License
Intended for demonstration purposes. Ensure proper licensing for models or data in production deployments.

## Acknowledgments

- UIDAI for Aadhaar-related information
- Hugging Face for transformer models
- Facebook Research for FAISS
- Gradio team for the UI framework
