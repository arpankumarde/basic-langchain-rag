from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
# from qdrant_client.http.models import Distance, VectorParams
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

client = QdrantClient(path="./qdb")
embeddings = FastEmbedEmbeddings()

# client.get_collection(collection_name="demo_collection")

client.create_collection(
    collection_name="demo_collection",
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="demo_collection",
    embedding=embeddings,
)
