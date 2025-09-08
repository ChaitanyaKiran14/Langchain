from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter,Language
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")


video_id = "106DaA8WdHg" 
ytt_api = YouTubeTranscriptApi()
vals = ytt_api.fetch(video_id)
full_transcript = " ".join([snippet.text for snippet in vals])

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=1000,
    chunk_overlap=150,
)
chunks = splitter.create_documents([full_transcript])
#print(chunks)
print(len(chunks))


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)

vectorStore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings,
)
#print(vectorStore.index_to_docstore_id)
#print(len(vectorStore.index_to_docstore_id))


#embed_val = vectorStore.get_by_ids('85fa3643-3ecb-46cf-a48a-0f100f1f806c')
#print(embed_val)

retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
#result = retriever.invoke("can you summarize")
#print(result)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)
question          = "in which year unemployment among recent college graduates reached 5.8%"
retrieved_docs    = retriever.invoke(question)

#print(retrieved_docs)
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
final_prompt = prompt.invoke({"context": context_text, "question": question})
answer = model.invoke(final_prompt)
print(answer.content)


#chain forming...

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parallel_chain.invoke('in which year unemployment among recent college graduates reached 5.8%')
parser = StrOutputParser()
main_chain = parallel_chain | prompt | model | parser
final_result = main_chain.invoke('Can you summarize the video')
print(final_result)