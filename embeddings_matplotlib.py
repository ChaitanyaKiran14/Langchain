from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L6-v2")

# Example texts
texts = [
    "Hey there",
    "Hello!",
    "How are you?",
    "What's up?",
    "See you later",
    "Banana",
    "Let's meet tomorrow",
    "Machine learning is great",
    "Good night",
    "Thermodynamics",
    "1st law of gases",
    "Physics"
]

# Get embeddings for each text
vectors = [embeddings.embed_query(text) for text in texts]

# Reduce to 2D using PCA
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], color='blue')

# Annotate each point with the text
for i, text in enumerate(texts):
    plt.annotate(text, (reduced_vectors[i, 0], reduced_vectors[i, 1]), fontsize=9)

plt.title("2D Visualization of Sentence Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()
