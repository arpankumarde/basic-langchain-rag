from uuid import uuid4
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


embeddings = OllamaEmbeddings(model="all-minilm")
vector_store = Chroma(
    collection_name="testing",
    embedding_function=embeddings,
    persist_directory="./db",
)

document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
    id=1,
)

document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
    id=2,
)

documents = [document_1, document_2]
doc_ids = [str(uuid4()) for _ in range(len(documents))]

vector_store.add_documents(documents=documents, ids=doc_ids)
