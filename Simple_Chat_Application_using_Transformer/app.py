import streamlit as st
import re
import json
import os


# db = {
#     "what is your return policy": ["Return", "within", "30", "days"],
#     "how can i track my order": ["Use", "tracking", "link"],
#     "do you offer cash on delivery": ["Yes", "available"],
#     "how long does delivery take": ["3", "to", "5", "days"],
#     "how to cancel my order": ["Go", "to", "My", "Orders"],
#     "do you ship internationally": ["Yes", "we", "do"],
#     "what payment methods are accepted": ["Cards,", "UPI,", "Netbanking"],
#     "can i change my delivery address": ["Yes,", "before", "shipping"],
#     "is my payment secure": ["Yes,", "fully", "secure"],
#     "do you offer discounts": ["Yes,", "check", "offers"]
# }

DB_FILE = "qa_memory.json"

# Load DB from file
if os.path.exists(DB_FILE):
    with open(DB_FILE, "r") as f:
        db = json.load(f)
else:
    db = {}

# Save DB to file
def save_db():
    with open(DB_FILE, "w") as f:
        json.dump(db, f)

word_embeddings = {
    "what": 0.2,
    "is": 0.3,
    "your": 0.3,
    "return": 0.9,
    "policy": 0.8,
    "how": 0.5,
    "can": 0.4,
    "i": 0.4,
    "track": 0.8,
    "my": 0.4,
    "order": 0.9,
    "do": 0.5,
    "you": 0.5,
    "offer": 0.7,
    "cash": 0.8,
    "on": 0.2,
    "delivery": 0.9,
    "long": 0.6,
    "does": 0.5,
    "take": 0.4,
    "to": 0.3,
    "cancel": 0.8,
    "ship": 0.9,
    "internationally": 0.7,
    "payment": 0.9,
    "methods": 0.7,
    "are": 0.3,
    "accepted": 0.6,
    "change": 0.8,
    "address": 0.8,
    "secure": 0.8,
    "discounts": 0.8
}

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def tokenize(text):
    return normalize(text).split()

def attention_score(w1, w2):
    return word_embeddings.get(w1, 0.1) * word_embeddings.get(w2, 0.1)

def encode_question(ques):
    tokens = tokenize(ques)
    focus_map = {}
    for w1 in tokens:
        total = 0.0
        for w2 in tokens:
            if w1 != w2:
                total += attention_score(w1, w2)
        focus_map[w1] = round(total, 3)
    return tokens, focus_map

def find_best_match(tokens):
    token_set = set(tokens)
    best_key = None
    best_score = -1.0
    for x in db:
        x_set = set(tokenize(x))
        intersection = token_set & x_set
        union = token_set | x_set
        score = len(intersection) / len(union) if union else 0.0
        if score > best_score:
            best_score = score
            best_key = x
    return best_key, best_score

def decoder(memory_key):
    if memory_key is None:
        return ["Answer", "not", "found!"]
    return db.get(memory_key, ["Answer", "not", "found!"])




# --- Streamlit App ---
st.title("Mini Transformer Chat")

user_input = st.text_input("Ask a question:")

if user_input:
    tokens, focus_map = encode_question(user_input)
    best_key, score = find_best_match(tokens)
    if score < 0.1:
        best_key = None
    answer = " ".join(decoder(best_key))

    st.subheader("Encoder")
    st.write(f"**Tokens:** {tokens}")
    st.write(f"**Focus Map:** {focus_map}")

    st.subheader("Decoder")
    st.write(f"**Best Match:** `{best_key}` (Score: {score:.2f})")
    st.write(f"**Answer:** {answer}")


st.markdown("---")
st.header("➕ Add New Q&A to Memory")

new_question = st.text_input("New Question:")
new_answer = st.text_input("Answer for this Question:")

if st.button("Add to Memory"):
    if new_question and new_answer:
        clean_q = normalize(new_question)
        db[clean_q] = tokenize(new_answer)
        save_db()  # save to file
        st.success("New question added to memory.")
    else:
        st.error("Both question and answer are required.")
