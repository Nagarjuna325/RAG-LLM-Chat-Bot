import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import openai

from dotenv import load_dotenv
load_dotenv()
## load the GROQ API Key
os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

groq_api_key=os.getenv("GROQ_API_KEY")

# creating the LLM model

llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

# creating the chat prompt template
# also we will we providing some context for the prompt how to responsed
prompt=ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question
    <context>
    {context}
    <context>
    Question:{input}

    """

)

# this function we will use to read the documents and store the data in some vector database.
# we also session state which is in particular with vectordb to have some memory.
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OpenAIEmbeddings() # we can also use Ollama but it takes time as it will be runnin locally on our machine so we use OpenAI embeddings
        st.session_state.loader=PyPDFDirectoryLoader("researchpapers") ## Data Ingestion step

        st.session_state.docs=st.session_state.loader.load() ## Document Loading

        # Debug: Check if documents are loaded
        # if not st.session_state.docs:
            #st.write("No documents were loaded. Please check the 'research_papers' directory.")
            #return
        #else:
            #st.write(f"Loaded {len(st.session_state.docs)} documents.")
            #for i, doc in enumerate(st.session_state.docs[:5]):  # Display the first 5 documents
                #st.write(f"Document {i + 1}: {doc.page_content[:200]}...")  # Show the first 200 characters of each doc

        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        
        # Debug: Check if documents were split
        #if not st.session_state.final_documents:
            #st.write("Documents were loaded but could not be split into chunks. Please check the text splitter settings.")
            #return
        #else:
            #st.write(f"Split into {len(st.session_state.final_documents)} chunks.")

        # storing the data in vector database, we also mentioning the embedding tehnique we use
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

        st.write("Vector database successfully created!")

st.title("RAG Document Q&A With Groq And Lama3")

user_prompt=st.text_input("Enter your query from the research paper")

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database is ready")

import time

if user_prompt:
    if "vectors" not in st.session_state or st.session_state.vectors is None:
        st.warning("Please create the vector database first by clicking on 'Document Embedding'.")
    else:
        # creates a llm 
        document_chain=create_stuff_documents_chain(llm,prompt)
        # if we want to get anything from vectordb we need to create a retriever to query a db
        retriever=st.session_state.vectors.as_retriever()
        # combaining the both document chain and retriever chain
        retrieval_chain=create_retrieval_chain(retriever,document_chain)
        # started the timer 
        start=time.process_time()
         # invoking the chain with user's input  and getting the response from the LLM and Vectordb.  This is the main function that answers the user's query.  The context here is the documents retrieved from the vector database.  The answer is the most relevant document from the vector database.  The response time is also calculated here.  The user can expand this expander to see the full document content.  This feature is useful to understand the context and the relevance
        response=retrieval_chain.invoke({'input':user_prompt})
        # calculating the response time
        print(f"Response time :{time.process_time()-start}")

        st.write(response['answer'])

       ## With a streamlit expander
        with st.expander("Document similarity Search"):
           for i,doc in enumerate(response['context']):
               st.write(doc.page_content)
               st.write('------------------------')
