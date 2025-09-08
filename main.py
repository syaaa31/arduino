import os
import requests
import faiss
import pickle
from flask import Flask, request
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# Flask app
app = Flask(__name__)

# 🔑 Keys & URLs
BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

# 🧠 Groq client
client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

# 🔤 Embedding model (local, free to use)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 📂 Paths
CHAPTERS_FOLDER = "notes"   # folder containing chapter1.txt, chapter2.txt...
INDEX_FILE = "faiss_index.pkl"


# === Utility: Load chapters and split into chunks ===
def load_and_chunk_notes():
    documents = []
    for filename in os.listdir(CHAPTERS_FOLDER):
        if filename.endswith(".txt"):
            chapter = filename.replace(".txt", "")
            with open(os.path.join(CHAPTERS_FOLDER, filename), "r", encoding="utf-8") as f:
                text = f.read()

            # split into ~200-word chunks
            words = text.split()
            chunk_size = 200
            chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

            for idx, chunk in enumerate(chunks):
                documents.append({
                    "chapter": chapter,
                    "chunk_id": idx,
                    "text": chunk
                })
    return documents


# === Utility: Build FAISS index ===
def build_faiss_index(docs):
    texts = [doc["text"] for doc in docs]
    embeddings = embedder.encode(texts, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, docs


# === Load or create FAISS ===
if os.path.exists(INDEX_FILE):
    with open(INDEX_FILE, "rb") as f:
        index, documents = pickle.load(f)
else:
    documents = load_and_chunk_notes()
    index, documents = build_faiss_index(documents)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump((index, documents), f)


# === Search function ===
def retrieve_relevant_chunks(query, k=3):
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
