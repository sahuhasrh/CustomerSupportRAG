
import numpy as np
from sentence_transformers import SentenceTransformer
from fastembed import TextEmbedding

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

text = "How can I reset my account password?"

print("Loading original model...")
original_model = SentenceTransformer(MODEL, device="cpu")
original = original_model.encode(
    text,
    normalize_embeddings=True
)

print("Loading FastEmbed model...")
fast_model = TextEmbedding(model_name=MODEL)
fast = np.array(list(fast_model.embed([text]))[0])

similarity = np.dot(original, fast) / (
    np.linalg.norm(original) * np.linalg.norm(fast)
)

print("\nOriginal dimensions:", original.shape)
print("FastEmbed dimensions:", fast.shape)
print("Cosine similarity:", similarity)

if original.shape == fast.shape and similarity > 0.999:
    print("\nPASS: Embeddings are highly compatible.")
else:
    print("\nCHECK: Do not replace the production retriever yet.")
