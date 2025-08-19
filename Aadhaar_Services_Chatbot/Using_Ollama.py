import streamlit as st
from streamlit_chat import message
import ollama
import pandas as pd
import numpy as np
import re
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer
import faiss
import unicodedata
from difflib import SequenceMatcher
import os
from PIL import Image
import base64
import json
from streamlit.components.v1 import html
import random

# === Configuration ===
MODEL_NAME = "llama3"  # or "mistral", "gemma", etc. llama3,llama2-7b
EMBED_MODEL = "all-MiniLM-L6-v2"
FAQ_CSV = "aadhaar_dataset_final1_upd.csv"
LOGO_PATH = "aadhaar_bg.jpg"  # Replace with your logo path
BG_COLOR = "#f8f9fa"  # Light gray background

# === Helper Functions (unchanged from original) ===
def load_data():
    """Load FAQ data and create embeddings"""
    faq_df = pd.read_csv(FAQ_CSV)
    embed_model = SentenceTransformer(EMBED_MODEL)
    faq_embeddings = embed_model.encode(faq_df['question'].tolist(), convert_to_numpy=True)
    
    # FAISS index
    d = faq_embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    faiss_index = faiss.IndexIVFFlat(quantizer, d, min(100, faq_embeddings.shape[0]))
    faiss_index.train(faq_embeddings)
    faiss_index.add(faq_embeddings)
    faiss_index.nprobe = 3
    
    return faq_df, embed_model, faiss_index

def normalize_text(text):
    text = text.lower()
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\b(aadhaar|aadhar|adhaar|adhhar)\b', 'aadhaar', text)
    return text.strip()

def retrieve_faq(query, faq_df, embed_model, faiss_index, k=3):
    query_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = faiss_index.search(query_emb, k)
    results = []
    for idx, score in zip(I[0], D[0]):
        if idx >= 0:
            results.append({
                "question": faq_df.iloc[idx]['question'],
                "answer": faq_df.iloc[idx]['answer'],
                "official_link": faq_df.iloc[idx]['official_link'] if pd.notna(faq_df.iloc[idx]['official_link']) else None,
                "score": float(1 - score)
            })
    return sorted(results, key=lambda x: x['score'], reverse=True)

# === Conversation Memory (unchanged from original) ===
class ConversationMemory:
    def __init__(self, max_turns=6):
        self.max_turns = max_turns
        self.turns = []
        self.last_topic = None
        self.last_intent = None

    def update(self, user_text, bot_text, topic=None, intent=None):
        if topic:
            self.last_topic = topic
        if intent:
            self.last_intent = intent
        self.turns.append({
            "user": user_text,
            "bot": bot_text,
            "topic": self.last_topic,
            "intent": self.last_intent,
            "ts": datetime.now().isoformat()
        })
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def resolve_followup(self, user_text):
        text = user_text.lower()
        anaphora = any(w in text for w in [" it", " that", " this", " them", " they"])
        generic_docs_q = re.search(r"\b(documents|docs|proof)\b", text) and not re.search(r"\b(address|mobile|phone|name|dob|date of birth|gender|biometric|pvc)\b", text)
        generic_fee_q = re.search(r"\b(cost|fee|charges?)\b", text) and not re.search(r"\b(address|mobile|phone|name|dob|date of birth|gender|biometric|pvc)\b", text)

        stated_topic, _ = detect_topic_and_intent(user_text)
        if stated_topic:
            return stated_topic, None
        if anaphora or generic_docs_q or generic_fee_q:
            return self.last_topic, self.last_intent
        return None, None

# === Topic & Intent Detection (unchanged from original) ===
TOPIC_KEYWORDS = {
    "address_update": ["address", "residence", "location"],
    "mobile_update": ["mobile", "phone", "number", "cell"],
    "name_update": ["name", "spelling", "rename"],
    "dob_update": ["dob", "date of birth", "birth"],
    "gender_update": ["gender", "sex"],
    "biometric_lock": ["biometric", "fingerprint", "iris", "lock", "unlock"],
    "pvc_card": ["pvc", "plastic", "card"],
}

INTENT_KEYWORDS = {
    "how_to": ["how", "procedure", "process", "steps", "update", "change"],
    "documents": ["document", "documents", "docs", "proof", "poa", "poi", "por", "dob"],
    "fees": ["cost", "fee", "charges", "price", "rs", "₹"],
    "status": ["status", "track", "tracking"],
}

def detect_topic_and_intent(text: str):
    t = text.lower()
    topic = None
    intent = None

    topic_hits = []
    for tname, kws in TOPIC_KEYWORDS.items():
        if any(kw in t for kw in kws):
            topic_hits.append(tname)
    if topic_hits:
        topic = topic_hits[0]

    for iname, kws in INTENT_KEYWORDS.items():
        if any(kw in t for kw in kws):
            intent = iname
            break

    return topic, intent

def rewrite_query_with_memory(user_text: str, mem: ConversationMemory):
    topic_map_to_human = {
        "address_update": "address update",
        "mobile_update": "mobile update",
        "name_update": "name update",
        "dob_update": "date of birth update",
        "gender_update": "gender update",
        "biometric_lock": "biometric lock/unlock",
        "pvc_card": "PVC card",
    }

    stated_topic, stated_intent = detect_topic_and_intent(user_text)
    mem_topic, mem_intent = mem.resolve_followup(user_text)

    topic = stated_topic or mem_topic
    intent = stated_intent or mem_intent

    needs_topic = re.search(r"\b(documents|docs|proof|fee|cost|charges?)\b", user_text.lower())
    if needs_topic and not topic:
        return user_text

    if topic:
        human_topic = topic_map_to_human.get(topic, topic.replace("_", " "))
        if re.search(r"\b(documents|docs|proof)\b", user_text.lower()) or intent == "documents":
            return f"What documents are required for {human_topic}?"
        if re.search(r"\b(cost|fee|charges?)\b", user_text.lower()) or intent == "fees":
            return f"What is the fee for {human_topic}?"
        if re.search(r"\bhow\b|\bsteps?\b|\bprocess\b|\bprocedure\b|\bupdate\b", user_text.lower()) or intent == "how_to":
            return f"How to do {human_topic}?"
        if re.search(r"\bstatus|track\b", user_text.lower()) or intent == "status":
            return f"How to check status for {human_topic}?"

    return user_text

# === Chatbot Function (unchanged from original) ===
def generate_response(user_input, faq_df, embed_model, faiss_index, mem):
    # Security check
    SENSITIVE_PATTERNS = [
        r"\bshare\b.*\bOTP\b",
        r"\benter\b.*\bOTP\b",
        r"\bgive\b.*\bOTP\b",
        r"\bsend\b.*\bOTP\b",
        r"\bdisclose\b.*\bOTP\b",
        r"\bfull\s*aadhaar\s*number\b",
        r"\bcard\s*details\b",
        r"\bpassword\b",
        r"\bCVV\b",
    ]
    SENSITIVE_RE = re.compile("|".join(SENSITIVE_PATTERNS), re.IGNORECASE)
    
    if SENSITIVE_RE.search(user_input or ""):
        return """**⚠️ Security Notice:**  
Never share OTPs, passwords, or full Aadhaar numbers here.  
Contact UIDAI helpline **1947** if you suspect fraud."""

    # Topic detection
    stated_topic, stated_intent = detect_topic_and_intent(user_input)
    mem_topic, mem_intent = mem.resolve_followup(user_input)
    
    # Query rewriting
    if mem_topic and not stated_topic:
        enriched_query = rewrite_query_with_memory(user_input, mem)
    else:
        enriched_query = user_input

    # Retrieve FAQs
    retrieved = retrieve_faq(enriched_query, faq_df, embed_model, faiss_index, k=3)

    # Generate prompt for Ollama
    context_parts = []
    for i, r in enumerate(retrieved[:2], 1):
        context = f"**Q:** {r['question']}\n**A:** {r['answer'][:300]}"
        if r.get('official_link'):
            context += f"\n[Official Link]({r['official_link']})"
        context_parts.append(context)
    
    context_text = "\n\n".join(context_parts)

    prompt = f"""You are an official Aadhaar Services assistant. Answer concisely using the context below.

    **Context:**
    {context_text}

    **User Question:** {user_input}

    **Assistant Answer:"""

    # Generate response with Ollama
    response = ollama.generate(
        model=MODEL_NAME,
        prompt=prompt,
        options={
            'temperature': 0.3,
            'num_predict': 150,
            'repeat_penalty': 1.2
        }
    )
    
    answer = response['response'].strip()
    
    # Add official link if not already in answer
    links = [r.get("official_link") for r in retrieved if r.get("official_link")]
    if links and "http" not in answer:
        answer += f"\n\n**Official Link:** {links[0]}"
    
    # Update memory
    resolved_topic, resolved_intent = detect_topic_and_intent(enriched_query)
    mem.update(user_input, answer, topic=resolved_topic, intent=resolved_intent)
    
    return answer

# === UI Functions ===
def add_custom_css():
    custom_css = """
    <style>
    /* Main container */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Header styling */
    .header {
        color: #003366;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 1.5rem;
    }
    
    /* Chat bubbles */
    .user-message {
        background: #0078d4;
        color: white;
        border-radius: 18px 18px 0 18px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .bot-message {
        background: white;
        color: #333;
        border-radius: 18px 18px 18px 0;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 80%;
        margin-right: auto;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    
    /* Input field */
    .stTextInput>div>div>input {
        width: 100% !important;
        border-radius: 25px;
        padding: 12px 20px;
        border: 1px solid #e0e0e0;
        font-size: 16px;
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 25px;
        background: #0078d4;
        color: white;
        padding: 10px 20px;
        border: none;
        font-weight: 500;
    }
    
    /* Cards */
    .feature-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        transition: all 0.2s;
    }
    
    .feature-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Quick links */
    .quick-link {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    
    /* Suggested questions */
    .suggested-question {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 20px;
        padding: 8px 16px;
        margin: 5px 0;
        cursor: pointer;
        font-size: 14px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .suggested-question:hover {
        background: #f0f7ff;
        border-color: #0078d4;
    }
    
    /* Chat container */
    .chat-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        margin-top: 2rem;
        color: #666;
        font-size: 0.9rem;
        border-top: 1px solid #e0e0e0;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    st.markdown(
    """
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/7.0.0/css/all.min.css" 
          rel="stylesheet">
    """,
    unsafe_allow_html=True
)

def create_header():
    st.markdown("""
    <div class="header">
        <h1 style="margin-bottom: 0.5rem; color: #003366;">Aadhaar Services Assistant</h1>
        <p style="color: #666;">Get official information about Aadhaar services</p>
    </div>
    """, unsafe_allow_html=True)

def create_quick_links():
    st.markdown("""
    <div style="display: flex; justify-content: space-between; margin: 1.5rem 0;">
        <a href="https://myaadhaar.uidai.gov.in/en_IN" target="_blank" style="text-decoration: none; flex: 1;">
            <div class="quick-link">
                <i class="fas fa-globe" style="font-size: 24px; color: #0078d4;"></i>
                <p style="margin: 8px 0 0; font-size: 14px;">UIDAI Website</p>
            </div>
        </a>
        <a href="https://myaadhaar.uidai.gov.in/en_IN" target="_blank" style="text-decoration: none; flex: 1;">
            <div class="quick-link">
                <i class="fas fa-mobile-alt" style="font-size: 24px; color: #0078d4;"></i>
                <p style="margin: 8px 0 0; font-size: 14px;">MyAadhaar</p>
            </div>
        </a>
        <a href="https://myaadhaar.uidai.gov.in/genricDownloadAadhaar/en" target="_blank" style="text-decoration: none; flex: 1;">
            <div class="quick-link">
                <i class="fas fa-download" style="font-size: 24px; color: #0078d4;"></i>
                <p style="margin: 8px 0 0; font-size: 14px;">e-Aadhaar</p>
            </div>
        </a>
        <a href="https://myaadhaar.uidai.gov.in/check-aadhaar" target="_blank" style="text-decoration: none; flex: 1;">
            <div class="quick-link">
                <i class="fas fa-search" style="font-size: 24px; color: #0078d4;"></i>
                <p style="margin: 8px 0 0; font-size: 14px;">Check Status</p>
            </div>
        </a>
    </div>
    """, unsafe_allow_html=True)

def create_feature_cards():
    features = [
        {"icon": "📝", "title": "Document Guidance", "desc": "Get document requirements for Aadhaar services"},
        {"icon": "💰", "title": "Fee Calculator", "desc": "Know charges for different services"},
        {"icon": "⏱️", "title": "Status Tracking", "desc": "Track your update requests"},
        {"icon": "🔒", "title": "Security Tips", "desc": "Aadhaar security guidelines"}
    ]
    
    st.markdown("### Quick Access")
    
    # Show 2 cards per row
    for i in range(0, len(features), 2):
        cols = st.columns(2)
        for j, feature in enumerate(features[i:i+2]):
            with cols[j]:
                st.markdown(f"""
                <div class="feature-card">
                    <div style="font-size: 24px; text-align: center; margin-bottom: 8px;">{feature['icon']}</div>
                    <h4 style="margin: 0 0 5px 0; text-align: center; font-size: 16px;">{feature['title']}</h4>
                    <p style="margin: 0; text-align: center; font-size: 14px; color: #666;">{feature['desc']}</p>
                </div>
                """, unsafe_allow_html=True)

def create_suggested_questions():
    questions = [
        "How to update mobile number in Aadhaar?",
        "What documents are needed for name change?",
        "How to download e-Aadhaar?",
        "What is the fee for PVC card?"
    ]
    
    st.markdown("### Common Questions")
    for q in questions:
        if st.button(q, key=f"suggest_{q}", use_container_width=True):
            st.session_state.past.append(q)
            with st.spinner("Getting information..."):
                response = generate_response(q, *st.session_state.faq_data, st.session_state.memory)
            st.session_state.generated.append(response)
            st.rerun()

def create_chat_history_sidebar():
    if st.session_state.generated:
        st.sidebar.markdown("### Chat History")
        for i in range(len(st.session_state.generated)):
            st.sidebar.markdown(f"""
            <div style="margin-bottom: 1rem; padding: 0.5rem; background: white; border-radius: 8px; border: 1px solid #e0e0e0;">
                <p style="margin: 0; font-weight: 500; font-size: 14px;">{st.session_state.past[i][:50]}...</p>
                <p style="margin: 0; font-size: 13px; color: #666;">{st.session_state.generated[i][:70]}...</p>
            </div>
            """, unsafe_allow_html=True)
        
        if st.sidebar.button("Clear History", use_container_width=True):
            st.session_state.generated = []
            st.session_state.past = []
            st.session_state.memory = ConversationMemory()
            st.rerun()

# def make_links_clickable(text: str) -> str:
#     """Convert plain URLs to clickable links, but ignore markdown [text](url) style links."""
#     # Ignore already formatted markdown links
#     if "[" in text and "](" in text:
#         return text  

#     url_pattern = re.compile(r'(https?://\S+)')
#     return url_pattern.sub(
#         r'<a href="\1" target="_blank" style="color:#0645AD; text-decoration: underline;">\1</a>',
#         text
#     )

def make_links_clickable(text: str) -> str:
    """Convert plain URLs to clickable links, but ignore markdown [text](url) style links."""
    # Ignore already formatted markdown links
    if "[" in text and "](" in text:
        return text  

    url_pattern = re.compile(r'(https?://\S+)')
    return url_pattern.sub(
        r'<a href="\1" target="_blank" style="color:#0645AD; text-decoration: underline;">\1</a>',
        text
    )

def render_markdown_table(text: str) -> str:
    """Convert markdown table to styled HTML table."""
    if "|" not in text:
        return text  # Not a table

    lines = text.strip().split("\n")
    headers = [h.strip() for h in lines[0].split("|")[1:-1]]
    rows = []
    for line in lines[2:]:  # skip header and separator
        row = [c.strip() for c in line.split("|")[1:-1]]
        if row:
            rows.append(row)

    html = '<table style="border-collapse: collapse; width: 100%; margin-top:5px;">'
    # Header
    html += '<tr style="background-color: #0078d4; color: white;">'
    for h in headers:
        html += f'<th style="padding: 8px; border: 1px solid #ddd; text-align:left;">{h}</th>'
    html += '</tr>'
    # Rows
    for idx, r in enumerate(rows):
        bg = '#f9f9f9' if idx % 2 == 0 else '#ffffff'
        html += f'<tr style="background-color:{bg};">'
        for cell in r:
            html += f'<td style="padding: 8px; border: 1px solid #ddd; text-align:left;">{make_links_clickable(cell)}</td>'
        html += '</tr>'
    html += '</table>'
    return html

def display_chat_messages():
    chat_html = """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.2.0/css/all.min.css"/>
    <div id="chat-container" 
         style="height: 400px; overflow-y: auto; margin-bottom: 1rem; 
                padding: 0.5rem; border: 1px solid #e0e0e0; 
                border-radius: 8px; background: #fafafa; font-family: Arial, sans-serif;">
    """

    for i in range(len(st.session_state.generated)):
        # User message (Right aligned)
        chat_html += f"""
        <div style="display: flex; justify-content: flex-end; margin-bottom: 10px;">
            <div style="background-color: #0078d4; color: white; padding: 8px 12px; 
                        border-radius: 12px 12px 0 12px; max-width: 70%; 
                        word-wrap: break-word; white-space: pre-line; display: flex; align-items: flex-start;">
                <div style="width: 28px; height: 28px; border-radius: 50%; background: white; 
                            display: flex; align-items: center; justify-content: center; margin-left: 8px; margin-right:6px;">
                    <i class="fa-solid fa-user" style="color: #0078d4; font-size: 14px;"></i>
                </div>
                <span>{st.session_state.past[i]}</span>
            </div>
        </div>
        """

        # Bot message (Left aligned)
        bot_msg = st.session_state.generated[i]
        bot_msg = render_markdown_table(bot_msg)  # render table if present
        bot_msg = make_links_clickable(bot_msg)

        chat_html += f"""
        <div style="display: flex; justify-content: flex-start; margin-bottom: 10px;">
            <div style="background-color: #e0e0e0; color: black; padding: 8px 12px; 
                        border-radius: 12px 12px 12px 0; max-width: 70%; 
                        word-wrap: break-word; white-space: pre-line; display: flex; align-items: flex-start; flex-direction: column;">
                <div style="width: 28px; height: 28px; border-radius: 50%; background: #0078d4; 
                            display: flex; align-items: center; justify-content: center; margin-right: 8px; margin-bottom:4px;">
                    <i class="fa-solid fa-robot" style="color: white; font-size: 14px;"></i>
                </div>
                <div>{bot_msg}</div>
            </div>
        </div>
        """

    chat_html += "</div>"

    # ✅ Auto-scroll
    chat_html += """
    <script>
    var chatBox = document.getElementById('chat-container');
    chatBox.scrollTop = chatBox.scrollHeight;
    </script>
    """

    st.components.v1.html(chat_html, height=450, scrolling=False)
    
def create_footer():
    st.markdown("""
    <div class="footer">
        <p>© 2025 UIDAI Aadhaar Services. All rights reserved.</p>
        <p style="font-size: 0.8rem; margin-top: 0.5rem;">
            <a href="#" style="color: #0078d4; text-decoration: none; margin: 0 10px;">Privacy Policy</a> | 
            <a href="#" style="color: #0078d4; text-decoration: none; margin: 0 10px;">Terms of Service</a> | 
            <a href="#" style="color: #0078d4; text-decoration: none; margin: 0 10px;">Contact Us</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

# === Main Streamlit UI ===
def main():
    # Initialize session state
    if 'generated' not in st.session_state:
        st.session_state.generated = []
    if 'past' not in st.session_state:
        st.session_state.past = []
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationMemory()
    if 'faq_data' not in st.session_state:
        with st.spinner("Loading Aadhaar knowledge base..."):
            st.session_state.faq_data = load_data()

    # Add custom CSS
    add_custom_css()
    
    # Sidebar
    with st.sidebar:
        try:
            logo = Image.open(LOGO_PATH)
            st.image(logo, use_container_width=True)
        except:
            st.image("https://upload.wikimedia.org/wikipedia/en/thumb/c/cf/Aadhaar_Logo.svg/1200px-Aadhaar_Logo.svg.png", use_container_width=True)
        
        st.markdown("""
        <div style="margin: 1.5rem 0;">
            <h3 style="color: #003366; margin-bottom: 0.5rem;">Aadhaar Services</h3>
            <p style="color: #666; font-size: 0.9rem;">Official information about Aadhaar services</p>
        </div>
        """, unsafe_allow_html=True)
        
        create_chat_history_sidebar()
        
        st.markdown("""
        <div style="margin-top: 2rem; padding: 1rem; background: #f0f7ff; border-radius: 8px;">
            <h4 style="color: #003366; margin-bottom: 0.5rem;">Need Help?</h4>
            <p style="color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;">Call UIDAI Helpline:</p>
            <p style="font-size: 1.2rem; font-weight: bold; color: #0078d4; margin: 0;">1947</p>
        </div>
        """, unsafe_allow_html=True)

    # Main content
    create_header()
    
    # --- Welcome things ---
    if not st.session_state.generated:
        create_quick_links()
        create_feature_cards()
        with st.sidebar:
            st.markdown("### Common Questions")
            create_suggested_questions()
    else:
        # Show chat messages when conversation has started
        display_chat_messages()


    # --- Chat input always visible ---
    with st.form(key='main_chat_form', clear_on_submit=True):
        user_input = st.text_input(
            "Ask about Aadhaar services:",
            key='chat_input',
            placeholder="Type your question here...",
            label_visibility="collapsed"
        )
        submit_button = st.form_submit_button("Send")

    if submit_button and user_input:
        st.session_state.past.append(user_input)
        with st.spinner("Searching Aadhaar knowledge base..."):
            response = generate_response(
                user_input,
                *st.session_state.faq_data,
                st.session_state.memory
            )
        st.session_state.generated.append(response)
        st.rerun()

    # --- Footer ---
    create_footer()


if __name__ == "__main__":
    main()