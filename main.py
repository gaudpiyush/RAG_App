import streamlit as st
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Your indexing imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# Your chatting imports
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="RAG PDF Chat Application",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'collection_ready' not in st.session_state:
    st.session_state.collection_ready = False

def index_pdf(uploaded_file):
    """Your indexing logic - processes PDF and creates vector embeddings"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        # Set up Google API key (from your indexing code)
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        
        # Loading (from your indexing code)
        loader = PyPDFLoader(file_path=tmp_path)
        docs = loader.load()  # Read PDF File
        
        # Chunking (from your indexing code)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=400
        )
        split_docs = text_splitter.split_documents(documents=docs)
        
        # Vector Embeddings (from your indexing code)
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
        
        # Using [embedding_model] create embeddings of [split_docs] and store in DB (from your indexing code)
        vector_store = QdrantVectorStore.from_documents(
            documents=split_docs,
            url="http://localhost:6333",
            collection_name="learning_vectors",
            embedding=embedding_model
        )
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return True, len(split_docs)
        
    except Exception as e:
        st.error(f"Error during indexing: {str(e)}")
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        return False, 0

def chat_with_pdf(query):
    """Your chatting logic - queries the indexed PDF"""
    try:
        # Configure Google Gemini API (from your chatting code)
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
        # Initialize the Gemini model (from your chatting code)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Vector Embeddings (from your chatting code)
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
        
        # Connect to existing collection (from your chatting code)
        vector_db = QdrantVectorStore.from_existing_collection(
            url="http://localhost:6333",
            collection_name="learning_vectors",
            embedding=embedding_model
        )
        
        # Vector Similarity Search [query] in DB (from your chatting code)
        search_results = vector_db.similarity_search(
            query=query
        )
        
        # Build context (from your chatting code)
        context = "\n\n\n".join([
            f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}" 
            for result in search_results
        ])
        
        # System prompt (from your chatting code)
        SYSTEM_PROMPT = f"""
            You are a helpful AI Assistant who answers user query based on the available context
            retrieved from a PDF file along with page_contents and page number.
            You should only answer the user based on the following context and navigate the user
            to open the right page number to know more.
            Context:
            {context}
        """
        
        # Create the full prompt for Gemini (from your chatting code)
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser Query: {query}"
        
        # Generate response using Gemini (from your chatting code)
        response = model.generate_content(full_prompt)
        
        return response.text
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Main App UI
st.title("📚 RAG PDF Chat Application")
st.markdown("*Upload a PDF, process it, and chat with its contents using Google Gemini AI*")
st.markdown("---")

# Sidebar for PDF upload and processing
with st.sidebar:
    st.header("📄 PDF Upload & Processing")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF file to chat with its contents"
    )
    
    # Process PDF button
    if uploaded_file is not None:
        st.success(f"📄 File uploaded: {uploaded_file.name}")
        st.info(f"📊 File size: {uploaded_file.size / 1024:.1f} KB")
        
        if st.button("🔄 Prepare this PDF", type="primary", use_container_width=True):
            with st.spinner("🔄 Processing PDF... This may take a few minutes."):
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("📖 Loading PDF...")
                progress_bar.progress(25)
                
                status_text.text("✂️ Splitting into chunks...")
                progress_bar.progress(50)
                
                status_text.text("🔤 Creating embeddings...")
                progress_bar.progress(75)
                
                # Call your indexing function
                success, num_chunks = index_pdf(uploaded_file)
                
                progress_bar.progress(100)
                
                if success:
                    st.session_state.pdf_processed = True
                    st.session_state.collection_ready = True
                    status_text.empty()
                    progress_bar.empty()
                    st.success(f"✅ PDF processed successfully!")
                    st.info(f"📊 Created {num_chunks} text chunks for searching")
                    st.balloons()  # Celebration effect
                else:
                    status_text.empty()
                    progress_bar.empty()
                    st.error("❌ Failed to process PDF")
                    st.session_state.pdf_processed = False
                    st.session_state.collection_ready = False
    
    # Status indicator
    st.markdown("---")
    st.subheader("🔧 System Status")
    
    if st.session_state.pdf_processed:
        st.success("🟢 Ready for Chatting!")
        st.info("💬 You can now ask questions about your PDF")
    else:
        st.warning("🟡 Upload and process a PDF to start chatting")
        st.info("👆 Please upload a PDF file first")
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# Main chat interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("💬 Chat with your PDF")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        if st.session_state.chat_history:
            for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
                # User message
                with st.chat_message("user"):
                    st.write(user_msg)
                
                # Bot message
                with st.chat_message("assistant"):
                    st.write(bot_msg)
        else:
            if st.session_state.pdf_processed:
                st.info("💡 Start by asking a question about your PDF!")
            else:
                st.info("👈 Upload and process a PDF file first to start chatting.")
    
    # Chat input
    if st.session_state.pdf_processed and st.session_state.collection_ready:
        user_query = st.chat_input("Ask a question about your PDF...")
        
        if user_query:
            # Add user message to chat
            with chat_container:
                with st.chat_message("user"):
                    st.write(user_query)
                
                # Generate and display response
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Thinking..."):
                        # Call your chatting function
                        response = chat_with_pdf(user_query)
                    st.write(response)
            
            # Save to history
            st.session_state.chat_history.append((user_query, response))
            st.rerun()
    
    elif not st.session_state.pdf_processed:
        st.chat_input("Upload and process a PDF first...", disabled=True)
    else:
        st.chat_input("Processing PDF, please wait...", disabled=True)

with col2:
    st.header("ℹ️ How to Use")
    st.markdown("""
    **Step-by-step guide:**
    
    1. **📤 Upload PDF**: 
       - Click "Choose a PDF file"
       - Select your PDF document
    
    2. **⚙️ Process**: 
       - Click "Prepare this PDF"
       - Wait for processing to complete
    
    3. **💬 Chat**: 
       - Ask questions about the PDF content
       - Get AI-powered answers with page references
    
    **✨ Features:**
    - 🔍 Semantic search through PDF
    - 📖 Page number references  
    - 💡 Context-aware responses
    - 🧠 Powered by Google Gemini AI
    - 🗄️ Vector database storage
    """)
    
    # Technical details
    if st.session_state.pdf_processed:
        st.markdown("---")
        st.markdown("**🔧 Technical Status:**")
        st.success("✅ Vector DB Connected")
        st.success("✅ Embeddings Ready") 
        st.success("✅ Gemini AI Active")
        st.success("✅ Chat Interface Ready")
        
        # Chat statistics
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown(f"**📊 Chat Stats:**")
            st.metric("Messages", len(st.session_state.chat_history))

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-style: italic;'>"
    "🚀 Built with Streamlit • 🤖 Powered by Google Gemini • 🗄️ Vector Search with Qdrant"
    "</div>", 
    unsafe_allow_html=True
)