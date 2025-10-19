from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings


embeddings = FastEmbedEmbeddings()
vector_store = Chroma(
    collection_name="testing",
    embedding_function=embeddings,
    persist_directory="./db",
)

query_embeddings = embeddings.embed_query("stock market")

results = vector_store.similarity_search_by_vector_with_relevance_scores(
    embedding=query_embeddings, k=1
)
for res, score in results:
    print(f"* [SIM={score:3f}] {res.page_content} [{res.metadata}]")
