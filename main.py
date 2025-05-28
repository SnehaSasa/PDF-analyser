import sys
import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings


def load_and_split_document(file_path):
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    return docs

def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def generate_answer(question, relevant_docs):
    context = "\n".join([doc.page_content for doc in relevant_docs]) 
    prompt = f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    generator = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base")
    output = generator(prompt, max_new_tokens=100)[0]['generated_text']

    return output.strip()

def main():
    if len(sys.argv) < 2:
        print("Usage: python rag_cli.py <file.txt|file.md>")
        return

    file_path = sys.argv[1]
    if not file_path.endswith((".txt", ".md")):
        print("❌ Only .txt or .md files are supported.")
        return
    if not os.path.exists(file_path):
        print("❌ File not found.")
        return

    print("📄 Loading and processing document...")
    chunks = load_and_split_document(file_path)
    vectorstore = build_vector_store(chunks)
    print("✅ Ready to chat with your document!\n")


    while True:
        query = input("Ask something (or type 'exit'): ").strip()
        if query.lower() == "exit":
            break

        relevant_docs = vectorstore.similarity_search(query, k=3)

        answer = generate_answer(query, relevant_docs)

        print("\n Best Answer:\n")
        print(f"[1] {answer}\n")

if __name__ == "__main__":
    main()
