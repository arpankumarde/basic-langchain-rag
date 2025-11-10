from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever, MultiQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import getpass
import os

load_dotenv()
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

# Add Tavily API key prompt if missing
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

embeddings = FastEmbedEmbeddings()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp", temperature=0.7, max_tokens=1500
)


# ============= VECTOR DB SETUP =============
def setup_vectordb(documents):
    """Initialize ChromaDB with financial documents"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="financial_docs",
        persist_directory="./chroma_db",  # v0.3 persistence
    )
    return vectordb


# ============= QUERY EXPANSION + RETRIEVAL =============
def create_rag_retriever(vectordb):
    """Create retriever with query expansion and reranking"""

    # Step 1: Multi-query retriever (generates 3 queries automatically)
    base_retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    multi_query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""Generate 3 diverse search phrases for retrieving financial documents.
        Focus on different aspects: technical terms, semantic meaning, and related concepts.
        
        Original query: {question}
        
        Generate exactly 3 alternative search phrases (one per line):""",
    )

    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=llm, prompt=multi_query_prompt
    )

    # Step 2: Flashrank reranker (LangChain wrapper)
    compressor = FlashrankRerank(
        model="ms-marco-TinyBERT-L-2-v2", top_n=5  # Final number after reranking
    )

    # Step 3: Compression retriever with deduplication + reranking
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=multi_query_retriever
    )

    return compression_retriever


# ============= DEFINE TOOLS =============
# Global retriever (will be set in main)
_retriever = None


@tool
def retrieve_financial_docs(query: str) -> str:
    """Retrieve relevant context from financial documents using RAG with query expansion and reranking"""
    global _retriever

    # Retriever handles: query expansion (3 queries) -> retrieval (5 per query)
    # -> deduplication -> reranking (top 5)
    docs = _retriever.invoke(query)

    context = "\n\n".join(
        [f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
    )

    return f"Retrieved Context:\n{context}"


@tool
def web_search(query: str) -> str:
    """Search the web for additional financial information when context is insufficient"""
    try:
        # Use TavilySearch tool and return results
        results = tavily_search_tool.invoke({"query": query})
        return str(results)
    except Exception as e:
        return f"Web search unavailable: {str(e)}"


# ============= LANGGRAPH AGENT STATE =============
class AgentState(TypedDict):
    """State for the agent graph"""

    messages: Annotated[Sequence[BaseMessage], add_messages]


# ============= AGENT NODES =============
def call_model(state: AgentState):
    """LLM node that decides whether to use tools or respond"""
    messages = state["messages"]
    response = llm.bind_tools([retrieve_financial_docs, web_search]).invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState):
    """Conditional edge: continue to tools or end"""
    last_message = state["messages"][-1]

    if not last_message.tool_calls:
        return "end"
    return "continue"


# ============= BUILD LANGGRAPH WORKFLOW =============
def create_rag_agent(retriever):
    """Create LangGraph agent with RAG and web search capabilities"""
    global _retriever
    _retriever = retriever

    # Define workflow
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode([retrieve_financial_docs, web_search]))

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edges
    workflow.add_conditional_edges(
        "agent", should_continue, {"continue": "tools", "end": END}
    )

    # Add edge from tools back to agent
    workflow.add_edge("tools", "agent")

    # Compile
    app = workflow.compile()
    return app


# ============= MAIN EXECUTION =============
def main():
    # Load documents
    # from langchain_community.document_loaders import PyPDFLoader

    # Example: Load your financial documents
    # docs = []
    # docs = PyPDFLoader("financial_report.pdf").load()

    # Setup vector database
    # vectordb = setup_vectordb(docs)
    vectordb = Chroma(
        embedding_function=embeddings,
        collection_name="financial_docs",
        persist_directory="./db",
    )

    # Create RAG retriever with query expansion + reranking
    rag_retriever = create_rag_retriever(vectordb)

    # Create LangGraph agent
    agent = create_rag_agent(rag_retriever)

    # System prompt for decision-making
    system_message = """You are a financial analysis assistant with access to two tools:

1. retrieve_financial_docs: Query the internal knowledge base (ALWAYS use this first)
2. web_search: Search the internet for additional information (only if internal docs insufficient)

WORKFLOW:
1. First, ALWAYS call retrieve_financial_docs with the user's query
2. Analyze if the retrieved context is sufficient
3. If sufficient: Generate comprehensive response
4. If insufficient: Call web_search for additional information
5. Synthesize all information into a detailed answer

Be thorough and accurate."""

    # Query the system
    query = "what was the net earnings of nvidia?"

    initial_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query},
    ]

    response = agent.invoke({"messages": initial_messages})

    # Extract final answer
    final_answer = response["messages"][-1].content
    print(final_answer)

    return response


if __name__ == "__main__":
    main()
