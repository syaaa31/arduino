import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Paths
CHAPTERS_FOLDER = "notes"   # folder with chapter1.txt, chapter2.txt...
INDEX_FILE = "faiss_index.pkl"

# Embedder (local only)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

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

# Build FAISS index
def build_faiss_index(docs):
    texts = [doc["text"] for doc in docs]
    embeddings = embedder.encode(texts, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, docs

if __name__ == "__main__":
    docs = load_and_chunk_notes()
    index, docs = build_faiss_index(docs)

    with open(INDEX_FILE, "wb") as f:
        pickle.dump((index, docs), f)

    print(f"✅ FAISS index built and saved to {INDEX_FILE} with {len(docs)} chunks.")
