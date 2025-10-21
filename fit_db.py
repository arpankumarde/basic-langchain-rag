import os
from datetime import datetime
from uuid import uuid4
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


embeddings = FastEmbedEmbeddings()
vector_store = Chroma(
    collection_name="test",
    embedding_function=embeddings,
    persist_directory="./db",
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def get_pdf_files(directory: str):
    """Get all PDF files from the specified directory."""
    pdf_files: list[str] = []
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            if filename.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(directory, filename))
    return pdf_files


def extract_text_from_pdf(pdf_path: str):
    """Extract text content from a PDF file."""
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text


# Get all PDF files from the data directory
data_directory = "./data"
pdf_files = get_pdf_files(data_directory)

documents: list[Document] = []
current_timestamp = datetime.now().isoformat()

# Process each PDF file
for pdf_path in pdf_files:
    filename = os.path.basename(pdf_path)
    print(f"Processing: {filename}")

    # Extract text from PDF
    text_content = extract_text_from_pdf(pdf_path)

    if text_content.strip():
        # Create a document with the full text
        doc = Document(
            page_content=text_content,
            metadata={"file": filename, "timestamp": current_timestamp},
        )

        # Split the document into chunks
        split_docs = text_splitter.split_documents([doc])

        # Update metadata for each chunk with chunk index
        for i, split_doc in enumerate(split_docs):
            split_doc.metadata = {
                "file": filename,
                "timestamp": current_timestamp,
                "chunk_index": i,
                "total_chunks": len(split_docs),
            }

        documents.extend(split_docs)
        print(f"  → Created {len(split_docs)} chunks from {filename}")

if documents:
    # Generate unique IDs for each document chunk
    doc_ids = [str(uuid4()) for _ in range(len(documents))]

    # Add documents to vector store
    vector_store.add_documents(documents=documents, ids=doc_ids)
    print(
        f"\nAdded {len(documents)} document chunks from {len(pdf_files)} PDF files to the vector store."
    )

    # Show chunk distribution per file
    file_chunks = {}
    for doc in documents:
        filename = doc.metadata["file"]
        file_chunks[filename] = file_chunks.get(filename, 0) + 1

    print("\nChunk distribution:")
    for filename, chunk_count in file_chunks.items():
        print(f"  {filename}: {chunk_count} chunks")

    # Test similarity search
    if documents:
        results = vector_store.similarity_search_with_score("content", k=3)
        print(f"\nSample search results:")
        for res, score in results:
            print(
                f"* [SIM={score:.3f}] File: {res.metadata['file']} (chunk {res.metadata['chunk_index']}/{res.metadata['total_chunks']})"
            )
            print(f"  Content: {res.page_content[:100]}...")
else:
    print("No PDF files found or no content extracted.")
