# from langchain.prompts import PromptTemplate # Prompt template
# from langchain.vectorstores import Chroma   # Store the vectors
# from langchain.text_splitter import RecursiveCharacterTextSplitter # Chunks
# from langchain_community.document_loaders import PyPDFLoader # Load the text
# from langchain.chains import VectorDBQA,RetrievalQA, LLMChain # Chains and Retrival ans
# from langchain.retrievers.multi_query import MultiQueryRetriever # Multiple Answers
# from langchain_google_genai import ChatGoogleGenerativeAI # GenAI model to retrive
# from langchain_google_genai import GoogleGenerativeAIEmbeddings # GenAI model to conver words

# from dotenv import google_api_key

# def load_data():
#     load=PyPDFLoader("budget_speech.pdf")
#     document=loader.load

# def split():
#     text_splitter=RecursiveCharacterTextSplitter(chunks_size=1000,chunk_overlap=0)
#     texts=text_splitter.split_document(load_data.document)

# def embedd():
#     embeddings=GoogleGenerativeAIEmbeddings(
#       model='models/embedding-001',
#       google_api_key=google_api_key,
#       task_type='RETRIEVAL_DOCUMENT')

# def vector():
#     vectordb = Chroma.from_documents(documents=split.texts, embedding=embedd.embeddings)

# def retrive():
#     retriver_from_llm= MultiQueryRetriever.from_llm(retrievervector.vectordb.as_retriever(search_kwargs={'k':5}),llm=chat_model) # Changed 'vecrordb' to 'vectordb' and 'search_kwards' to 'search_kwargs'


# def qa():
#     qa_chain= RetrievalQA.from_chain_type(llm=chat_model,chain_type='stuff',retriever=retriver_from_llm,chain_type_kwargs={'prompt':prompt})


# if __name__ == '__main__':
#     qa()

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


