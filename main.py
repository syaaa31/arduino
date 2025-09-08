import os
import requests
import faiss
import pickle
from flask import Flask, request
from openai import OpenAI

# Flask app
app = Flask(__name__)

# 🔑 Keys & URLs
BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

# 🧠 Groq client
client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

# 📂 Load precomputed FAISS index
INDEX_FILE = "faiss_index.pkl"
with open(INDEX_FILE, "rb") as f:
    index, documents = pickle.load(f)

# === Search function ===
def retrieve_relevant_chunks(query, k=3):
    # ⚠️ No embeddings generated here, only FAISS search
    import numpy as np
    from sentence_transformers import SentenceTransformer

    # Load embedder only for query
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    query_emb = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(documents):
            results.append(documents[idx]["text"])
    return results

# === Telegram Bot ===
@app.route(f"/webhook/{BOT_TOKEN}", methods=["POST"])
def webhook():
    data = request.get_json()

    if "message" in data:
        chat_id = data["message"]["chat"]["id"]
        user_text = data["message"].get("text", "")

        # 1️⃣ Retrieve context
        chunks = retrieve_relevant_chunks(user_text, k=3)
        context = "\n\n".join(chunks)

        # 2️⃣ Call Groq API
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful tutor. Answer only from the provided notes."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_text}"}
            ]
        )

        answer = response.choices[0].message.content

        # 3️⃣ Reply back
        payload = {
            "chat_id": chat_id,
            "text": answer
        }
        requests.post(TELEGRAM_API_URL, json=payload)

    return "ok", 200

@app.route("/")
def home():
    return "RAG Bot is running!", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
