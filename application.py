import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time

from dotenv import load_dotenv
load_dotenv()

# Load the NVIDIA API key from the environment variables
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

# Function to set up the application state for handling document processing and embeddings
def vector_embedding():
    # Check if vectors are already in session state
    if "vectors" not in st.session_state:
        # Initialize the NVIDIA embeddings
        st.session_state.embeddings = NVIDIAEmbeddings()
        # Load documents from the specified directory
        st.session_state.loader = PyPDFDirectoryLoader("./documents")
        st.session_state.docs = st.session_state.loader.load()
        # Split documents into chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:30])
        # Create FAISS vector store from the document chunks using embeddings
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Set the title of the Streamlit app
st.title("Demo app NVIDIA NIM-LLAMA3")

# Initialize the large language model (LLM) with NVIDIA's llama3-70b-instruct model
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

# Design prompt for the LLM using a template
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions:{input}
"""
)

# Ask user to input their question
prompt1 = st.text_input("Enter Your Question From Documents")

# Button to create the document embeddings and vector store database
if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

# If the user has entered a question
if prompt1:
    # Create a document chain with the LLM and the prompt
    document_chain = create_stuff_documents_chain(llm, prompt)
    # Create a retriever from the vector store
    retriever = st.session_state.vectors.as_retriever()
    # Create a retrieval chain with the retriever and document chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    # Measure the response time
    start = time.process_time()
    # Invoke the retrieval chain with the user's question
    response = retrieval_chain.invoke({'input': prompt1})
    print("Response time :", time.process_time() - start)
    # Display the response in the Streamlit app
    st.write(response['answer'])

    # With a Streamlit expander to show document similarity search results
    with st.expander("Document Similarity Search"):
        # Display each relevant chunk of the documents
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
