🧠 CLI-Based RAG System for Text Document QA


This is a CLI-based Retrieval-Augmented Generation (RAG) system that allows you to upload `.txt` or `.md` documents, ask natural language questions, and get answers grounded in the document content.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🛠️ Setup Instructions


1. Clone the Repository

   git clone https://github.com/SnehaSasa/PDF-analyser
   cd PDF-analyser

2. Create a Virtual Environment

   python -m venv venv
   ./myvenv/Scripts/activate

3. Install Dependencies

   pip install -r requirements.txt

4. Download Required Models (if not downloaded automatically)

   The system uses HuggingFace models like flan-t5-base and all-MiniLM-L6-v2. These will be downloaded automatically when run for the first time. Internet connection is required for the first run.



🚀 Usage

1. Run the RAG CLI

   python main.py Quantrail-Data.txt

2. Enter a question:

   Ask something (or type 'exit'): What is the notice period?

The system will retrieve the most relevant chunks from the document and generate a grounded answer using the LLM.



⚙️ Architecture Overview

This project implements a Retrieval-Augmented Generation pipeline with the following components:

1. Document Ingestion - Reads .txt or .md file using LangChain’s TextLoader.

2. Chunking - Uses RecursiveCharacterTextSplitter with a chunk size of 500 characters and 50 overlap.

3. Embeddings - Uses HuggingFace’s all-MiniLM-L6-v2 to convert chunks into dense vectors.

4. Vector Store - FAISS is used to store and search embeddings for top-k retrieval.

5. Retrieval - Finds the top-k chunks most similar to the user’s query.

6. Answer Generation - Passes the query and top chunks to flan-t5-base to generate a final answer.

🧪 Example

Input File: Quantrail-Data.txt

Query: "What is the Notice Period?"

Sample Output:
One Month



🖼️ Architecture Diagram

[Text File] 
   ↓
[Chunking]
   ↓
[Embeddings (HuggingFace)]
   ↓
[Vector Store (FAISS)]
   ↓
[User Query] → [Similarity Search]
                       ↓
                  [Prompt + Top-k Chunks]
                       ↓
                  [LLM (Flan-T5)] 
                       ↓
                  [Answer Output]
                  


📁 Project Structure

PDF-analyser/
│
├── main.py                    # Main code
├── requirements.txt           # Python dependencies
├── Quantrail-Data.txt         # Sample .txt File
└── README.md                  # Project documentation



⚠️ Limitations

1. Only .txt and .md files are supported.

2. No support for PDFs or docx in this version.

3. May hallucinate if relevant chunks aren't retrieved.

4. Runs locally — ideal for small to medium-sized documents.

5. LLM context limit can truncate long inputs.



🔧 Design Choices

1. LLM Chosen: flan-t5-base from HuggingFace (lightweight, performant)

2. Embedding Model: all-MiniLM-L6-v2 (balanced between speed and quality)

3. Vector Store: FAISS for fast similarity search

4. Modular Design: Easily swap out models or DB backends

5. Chunking Strategy: Recursive splitter to ensure semantic overlap



👤 Contributors

Sneha A – End-to-end development, architecture design, and CLI integration



📌 Future Improvements

1. Add PDF and docx support

2. Add web interface (Flask or Streamlit)

Switch to more powerful LLM (e.g., Mistral, GPT-J)

Include RAG confidence scores and sources in output

