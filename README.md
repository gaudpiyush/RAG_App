# 📚 RAG_App

A simple **Retrieval-Augmented Generation (RAG)** application built with **Streamlit**, **Google Gemini**, and **Qdrant**.  
It lets you upload a PDF, creates embeddings, stores them in a vector database, and allows you to **chat with the PDF content**.

---

# 🧠 What This App Does (Full Process)

## **1. Upload a PDF**
- User selects a PDF from their system.
- File gets temporarily saved.
- The app reads the PDF using `PyPDFLoader`.

## **2. Process the Document**
- Splits the PDF into text chunks using `RecursiveCharacterTextSplitter`.
- Generates embeddings using **Google Gemini (embedding-001)**.
- Stores all embeddings inside **Qdrant vector database**.
- Marks the document as "ready for chat."

## **3. Chat with the PDF**
- User asks a question.
- The query is converted to embeddings.
- Qdrant performs **semantic similarity search**.
- The top relevant chunks are selected as *context*.
- Gemini model generates a final answer **strictly based on the PDF content**.
- The chat UI displays:
  - User query  
  - Gemini response  
  - Page references for transparency  
  - Chat history

---

# 🛠️ Setup Guide

## **1. Clone the Repository**
```bash
git clone <your-repo-url>
cd RAG_App
```

---

## **2. Install Required Dependencies**
```bash
pip install -r requirements.txt
```

---

## ***3. Add Environment Variables***

Create .env file in project root:
```bash
GOOGLE_API_KEY=your_key_here
```
Make sure the key supports Gemini models.

---

## ***4. Start Qdrant (Vector Database)***
Docker Compose
```bash
docker compose up -d
```

Qdrant will now run at:
```bash
http://localhost:6333
```

---

## ***5. Run the Application***
```bash
streamlit run main.py
```

---

# 📄 Steps to use the app:

## 1. Upload a PDF
- Click "Choose a PDF file"
- File details appear in the sidebar

## 2. Process the PDF
- Click "Prepare this PDF"
- You will see a progress bar:
    - Loading PDF
    - Splitting into chunks
    - Creating embeddings
    - Storing in Qdrant
- On success:
    - Chunk count is shown
    - Vector collection becomes ready
    - App shows “Ready for Chatting!”

## 3. Ask Questions
- Type your question in the chat box
- The system:
    - Retrieves similar chunks
    - Builds a context block
    - Generates an answer using Gemini
    - Shows the result in chat
- Chat history is preserved
- You can clear chat any time

---

# 🔧 Tools Used
- **Streamlit** - for frontend UI and chat interface
- **LangChain** – PDF loading, chunking, and vector store integration  
- **Gemini 1.5 Flash** – AI assistant backend
- **Qdrant** – vector database for semantic search  
- **Python** – core backend language for the entire application  
- **dotenv** – for secure API key loading  
