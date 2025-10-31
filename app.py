
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
                model_name="Llama-3.1-8b-instant",
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