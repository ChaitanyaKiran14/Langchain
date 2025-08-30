from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

#define embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key="AIzaSyDwEvaA5x03T-Xrne82OC_eQplXGLBdAi0"
)

docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

#define vector store
vectorStore = FAISS.from_documents(
    documents=docs,
    embedding=embeddings,
)

#define retriver

retriver = vectorStore.as_retriever( search_type="mmr",
        search_kwargs={'k': 2, 'lambda_mult': 0.25})

query = "What is langchain?"
results = retriver.invoke(query)
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)