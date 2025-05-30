# import streamlit as st
# import os
# from langchain_groq import ChatGroq
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# import openai

# from dotenv import load_dotenv
# load_dotenv()
# ## load the GROQ API Key
# os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
# os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

# groq_api_key=os.getenv("GROQ_API_KEY")


# # creating the LLM model

# llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

# # creating the chat prompt template
# # also we will we providing some context for the prompt how to responsed
# prompt=ChatPromptTemplate.from_template(
#     """
#     Answer the questions based on the provided context only.
#     Please provide the most accurate respone based on the question
#     <context>
#     {context}
#     <context>
#     Question:{input}

#     """

# )

# # this function we will use to read the documents and store the data in some vector database.
# # we also session state which is in particular with vectordb to have some memory.
# def create_vector_embedding():
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings=OpenAIEmbeddings() # we can also use Ollama but it takes time as it will be runnin locally on our machine so we use OpenAI embeddings
#         st.session_state.loader=PyPDFDirectoryLoader("researchpapers") ## Data Ingestion step

#         st.session_state.docs=st.session_state.loader.load() ## Document Loading

#         # Debug: Check if documents are loaded
#         # if not st.session_state.docs:
#             #st.write("No documents were loaded. Please check the 'research_papers' directory.")
#             #return
#         #else:
#             #st.write(f"Loaded {len(st.session_state.docs)} documents.")
#             #for i, doc in enumerate(st.session_state.docs[:5]):  # Display the first 5 documents
#                 #st.write(f"Document {i + 1}: {doc.page_content[:200]}...")  # Show the first 200 characters of each doc

#         st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
#         st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        
#         # Debug: Check if documents were split
#         #if not st.session_state.final_documents:
#             #st.write("Documents were loaded but could not be split into chunks. Please check the text splitter settings.")
#             #return
#         #else:
#             #st.write(f"Split into {len(st.session_state.final_documents)} chunks.")

#         # storing the data in vector database, we also mentioning the embedding tehnique we use
#         st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

#         st.success("Vector database successfully created!")

# st.title("RAG Document Q&A With Groq And Lama3")

# # Clear flag handler – MUST BE BEFORE text_input is declared
# if st.session_state.get("clear_input_flag"):
#     st.session_state.clear_input_flag = False
#     if "user_prompt" in st.session_state:
#         del st.session_state["user_prompt"]
#     st.session_state["user_prompt"] = ""  # set it to empty string to clear input box visually
#     st.rerun()

# user_prompt=st.text_input("Enter your query from the research paper", key="user_prompt", value=st.session_state.get("user_prompt", ""))

# # if st.button("Submit"):
# #     create_vector_embedding()
# #     st.write("Vector Database is ready")
# # Create two columns for Submit and Clear buttons

# status_placeholder = st.empty()  # Add at top

# col1, spacer, col2 = st.columns([1, 6, 1])

# with col1:
#     if st.button("Submit"):
#         create_vector_embedding()
#         status_placeholder.success("Vector Database is ready")

# with col2:
#     if st.button("Clear"):
#         st.session_state["clear_input_flag"] = True  # Clear the prompt

# import time

# if user_prompt:
#     if "vectors" not in st.session_state or st.session_state.vectors is None:
#         st.warning("Please create the vector database first by clicking on 'Submit'.")
#     else:
#         # creates a llm 
#         document_chain=create_stuff_documents_chain(llm,prompt)
#         # if we want to get anything from vectordb we need to create a retriever to query a db
#         retriever=st.session_state.vectors.as_retriever()
#         # combaining the both document chain and retriever chain
#         retrieval_chain=create_retrieval_chain(retriever,document_chain)
#         # started the timer 
#         start=time.process_time()
#          # invoking the chain with user's input  and getting the response from the LLM and Vectordb.  This is the main function that answers the user's query.  The context here is the documents retrieved from the vector database.  The answer is the most relevant document from the vector database.  The response time is also calculated here.  The user can expand this expander to see the full document content.  This feature is useful to understand the context and the relevance
#         response=retrieval_chain.invoke({'input':user_prompt})
#         # calculating the response time
#         print(f"Response time :{time.process_time()-start}")

#         st.write(response['answer'])

#        ## With a streamlit expander
#         with st.expander("Document similarity Search"):
#            for i,doc in enumerate(response['context']):
#                st.write(doc.page_content)
#                st.write('------------------------')

import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

# Load API keys from environment variables
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")

groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize session state variables
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "is_db_created" not in st.session_state:
    st.session_state.is_db_created = False
if "clear_input_flag" not in st.session_state:
    st.session_state.clear_input_flag = False

# Creating the chat prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.

    <context>
    {context}
    </context>
    
    Question: {input}
    """
)

def create_vector_embedding():
    """Function to create vector embeddings from PDF documents"""
    with st.spinner("Creating vector database - please wait..."):
        try:
            # Check if directory exists
            if not os.path.exists("researchpapers"):
                st.error("Directory 'researchpapers' not found. Please create it and add PDF files.")
                return False
            
            # Check if directory has files
            if len([f for f in os.listdir("researchpapers") if f.endswith('.pdf')]) == 0:
                st.error("No PDF files found in 'researchpapers' directory.")
                return False
                
            # Use OpenAI embeddings
            st.session_state.embeddings = OpenAIEmbeddings()
            
            # Load documents
            st.session_state.loader = PyPDFDirectoryLoader("researchpapers")
            st.session_state.docs = st.session_state.loader.load()
            
            if not st.session_state.docs:
                st.error("No content loaded from PDF files. Please check your documents.")
                return False
                
            # Split documents
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                st.session_state.docs[:100]  # Increased from 50 to 100
            )
            
            # Create vector database
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents, 
                st.session_state.embeddings
            )
            
            st.session_state.is_db_created = True
            return True
            
        except Exception as e:
            st.error(f"Error creating vector database: {str(e)}")
            return False

def query_documents(user_prompt):
    """Function to query documents using the LLM and vector store"""
    try:
        # Create LLM - with error handling
        try:
            llm = ChatGroq(
                groq_api_key=groq_api_key, 
                model_name="Llama3-8b-8192",
                temperature=0.2,  # Lower temperature for more factual responses
                max_tokens=1024,  # Setting a reasonable max token limit
                timeout=60  # Add timeout to prevent hanging
            )
        except Exception as e:
            st.error(f"Error connecting to Groq API: {str(e)}")
            st.info("Check your API key and network connection. The Groq service might be experiencing issues.")
            return None

        # Create the document chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Create retriever
        retriever = st.session_state.vectors.as_retriever(
            search_kwargs={"k": 5}  # Retrieve top 5 most relevant documents
        )
        
        # Create retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        # Query with progress indicator
        with st.spinner("Generating response from LLM..."):
            import time
            start = time.process_time()
            response = retrieval_chain.invoke({'input': user_prompt})
            elapsed_time = time.process_time() - start
            
        return response, elapsed_time
        
    except Exception as e:
        st.error(f"Error querying documents: {str(e)}")
        return None, 0

# App UI
st.title("RAG Document Q&A With Groq And Llama3")
st.markdown("""
This app uses Retrieval-Augmented Generation (RAG) to answer questions based on your PDF documents.
""")

# Status placeholder
status_placeholder = st.empty()

# Input for user query
user_prompt = st.text_input(
    "Enter your query ", 
    key="user_prompt", 
    value=st.session_state.get("user_prompt", ""),
    help="Type your question here"
)

# Button layout
col1, spacer, col2 = st.columns([1, 6, 1])

with col1:
    if st.button("Submit"):
        if not st.session_state.is_db_created:
            success = create_vector_embedding()
            if success:
                status_placeholder.success("Vector Database is ready")
            else:
                status_placeholder.error("Failed to create Vector Database")
        else:
            status_placeholder.success("Vector Database is ready")

with col2:
    if st.button("Clear"):
        st.session_state.clear_input_flag = True
        st.rerun()

# Handle clearing input
if st.session_state.get("clear_input_flag"):
    st.session_state.clear_input_flag = False
    if "user_prompt" in st.session_state:
        del st.session_state["user_prompt"]
    st.session_state["user_prompt"] = ""
    st.rerun()

# Process query
if user_prompt:
    if not st.session_state.is_db_created or st.session_state.vectors is None:
        st.warning("Please create the vector database first by clicking on 'Submit'.")
    else:
        # Display a message while processing
        with st.spinner("Processing your query..."):
            result = query_documents(user_prompt)
            
            if result:
                response, elapsed_time = result
                
                # Display the answer
                st.subheader("Answer:")
                st.write(response['answer'])
                st.caption(f"Response time: {elapsed_time:.2f} seconds")

                # Display source documents
                with st.expander("Document similarity Search"):
                    for i, doc in enumerate(response['context']):
                        st.markdown(f"**Document {i+1}**")
                        st.write(doc.page_content)
                        st.write('------------------------')