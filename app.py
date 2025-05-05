from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from dotenv import load_dotenv
import os



# Load environment variables
load_dotenv()

def load_data():
    loader = PyPDFLoader("budget_speech.pdf")
    documents = loader.load()
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model='models/embedding-001',
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        task_type='RETRIEVAL_DOCUMENT')

def create_vectorstore(texts, embeddings):
    return Chroma.from_documents(documents=texts, embedding=embeddings)

def create_retriever(vectordb, llm):
    return MultiQueryRetriever.from_llm(
        retriever=vectordb.as_retriever(search_kwargs={'k':5}),
        llm=llm)

def create_qa_chain(llm, retriever, prompt):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        chain_type_kwargs={'prompt': prompt})

def vector():
    vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./chroma_db",
    client_settings=chromadb.config.Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./chroma_db"
    ))
# In your app.py replace Chroma with:


def create_vectorstore(texts, embeddings):
    return FAISS.from_documents(documents=texts, embedding=embeddings)

# Later to save/load:
# vectorstore.save_local("faiss_index")
# loaded_store = FAISS.load_local("faiss_index", embeddings)


