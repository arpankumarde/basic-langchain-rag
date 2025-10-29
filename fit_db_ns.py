import os
from datetime import datetime
from uuid import uuid4
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from time import perf_counter


embeddings = FastEmbedEmbeddings()
vector_store = Chroma(
    collection_name="t_ns",
    embedding_function=embeddings,
    persist_directory="./db",
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# ChromaDB batch size limit: 5461 - set to safe value
MAX_BATCH_SIZE = 5000


def get_pdf_files(directory: str):
    """Get all PDF files from the specified directory and subdirectories with namespace."""
    pdf_files: list[tuple[str, str]] = []  # [(pdf_path, namespace), ...]
    if os.path.exists(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                # This is a namespace folder (e.g., Amazon, Uber)
                namespace = item.lower()
                for filename in os.listdir(item_path):
                    if filename.lower().endswith(".pdf"):
                        pdf_path = os.path.join(item_path, filename)
                        pdf_files.append((pdf_path, namespace))
            elif item.lower().endswith(".pdf"):
                # PDF directly in data_ns folder (no namespace)
                pdf_files.append((item_path, "default"))
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


def add_documents_in_batches(
    documents: list[Document], doc_ids: list[str], batch_size: int = MAX_BATCH_SIZE
):
    """Add documents to vector store in batches to avoid batch size limits."""
    total_docs = len(documents)
    total_batches = (total_docs + batch_size - 1) // batch_size

    print(
        f"Adding {total_docs} documents in {total_batches} batches (max {batch_size} per batch)"
    )

    for i in range(0, total_docs, batch_size):
        batch_end = min(i + batch_size, total_docs)
        batch_docs = documents[i:batch_end]
        batch_ids = doc_ids[i:batch_end]

        batch_num = (i // batch_size) + 1
        print(
            f"  Processing batch {batch_num}/{total_batches}: {len(batch_docs)} documents"
        )

        try:
            vector_store.add_documents(documents=batch_docs, ids=batch_ids)
            print(f"  --> Batch {batch_num} completed successfully")
        except Exception as e:
            print(f"  --> Error in batch {batch_num}: {e}")
            # Continue with next batch instead of failing completely
            continue


def get_available_namespaces(documents: list[Document]):
    """Get unique namespaces from documents."""
    namespaces = set()
    for doc in documents:
        if "namespace" in doc.metadata:
            namespaces.add(doc.metadata["namespace"])
    return sorted(list(namespaces))


if __name__ == "__main__":
    # Get all PDF files from the data directory
    data_directory = "./data_ns"
    pdf_files = get_pdf_files(data_directory)

    documents: list[Document] = []
    current_timestamp = datetime.now().isoformat()

    # Process each PDF file
    for pdf_path, namespace in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"Processing: {filename} [namespace: {namespace}]")

        # Extract text from PDF
        text_content = extract_text_from_pdf(pdf_path)

        if text_content.strip():
            # Create a document with the full text
            doc = Document(
                page_content=text_content,
                metadata={
                    "file": filename,
                    "namespace": namespace,
                    "timestamp": current_timestamp,
                },
            )

            # Split the document into chunks
            split_docs = text_splitter.split_documents([doc])

            # Update metadata for each chunk with chunk index
            for i, split_doc in enumerate(split_docs):
                split_doc.metadata = {
                    "file": filename,
                    "namespace": namespace,
                    "timestamp": current_timestamp,
                    "chunk_index": i,
                    "total_chunks": len(split_docs),
                }

            documents.extend(split_docs)
            print(f"  --> Created {len(split_docs)} chunks from {filename}")

    if documents:
        print(f"\nTotal document chunks to add: {len(documents)}")
        # Generate unique IDs for each document chunk
        start_time = perf_counter()
        doc_ids = [str(uuid4()) for _ in range(len(documents))]
        end_time = perf_counter()

        print(
            f"  --> Generated {len(doc_ids)} IDs in {end_time - start_time:.4f} seconds"
        )

        # Add documents to vector store in batches
        add_documents_in_batches(documents, doc_ids)
        print(
            f"\nCompleted adding {len(documents)} document chunks from {len(pdf_files)} PDF files to the vector store."
        )

        # Show namespace distribution
        namespace_chunks = {}
        for doc in documents:
            namespace = doc.metadata.get("namespace", "unknown")
            namespace_chunks[namespace] = namespace_chunks.get(namespace, 0) + 1

        print("\nNamespace distribution:")
        for namespace, chunk_count in namespace_chunks.items():
            print(f"  {namespace}: {chunk_count} chunks")

        # Show chunk distribution per file
        file_chunks = {}
        for doc in documents:
            filename = doc.metadata["file"]
            file_chunks[filename] = file_chunks.get(filename, 0) + 1

        print("\nChunk distribution:")
        for filename, chunk_count in file_chunks.items():
            namespace = next(
                (
                    d.metadata["namespace"]
                    for d in documents
                    if d.metadata["file"] == filename
                ),
                "unknown",
            )
            print(f"  {filename} [{namespace}]: {chunk_count} chunks")

        # Get available namespaces
        available_namespaces = get_available_namespaces(documents)
        print(f"\nAvailable namespaces: {', '.join(available_namespaces)}")

        # Test similarity search with namespace filtering
        if documents and available_namespaces:
            # General search
            results = vector_store.similarity_search_with_score("content", k=3)
            print(f"\nSample search results (all namespaces):")
            for res, score in results:
                print(
                    f"* [SIM={score:.3f}] Namespace: {res.metadata['namespace']} | File: {res.metadata['file']} (chunk {res.metadata['chunk_index']}/{res.metadata['total_chunks']})"
                )
                print(f"  Content: {res.page_content[:100]}...")

            # Namespace-specific search example
            test_namespace = available_namespaces[0]
            results_filtered = vector_store.similarity_search_with_score(
                "content", k=3, filter={"namespace": test_namespace}
            )
            print(f"\nSample search results (namespace='{test_namespace}' only):")
            for res, score in results_filtered:
                print(
                    f"* [SIM={score:.3f}] File: {res.metadata['file']} (chunk {res.metadata['chunk_index']}/{res.metadata['total_chunks']})"
                )
                print(f"  Content: {res.page_content[:100]}...")
    else:
        print("No PDF files found or no content extracted.")
