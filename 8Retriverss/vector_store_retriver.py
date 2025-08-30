from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key="AIzaSyDwEvaA5x03T-Xrne82OC_eQplXGLBdAi0"
)
# Create Chroma vector store in memory
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
        collection_name="my_collection"
)
# Convert vectorstore into a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
query = "What is Chroma used for?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)


results = vectorstore.similarity_search(query, k=2)
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)