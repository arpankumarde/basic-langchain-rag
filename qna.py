import os
import getpass
from typing import List, Any
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.retrievers import ContextualCompressionRetriever

# from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

# Load environment variables
load_dotenv()
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")


class QnABot:
    def __init__(self, persist_directory: str = "./db", collection_name: str = "test"):
        """Initialize the QnA bot with ChromaDB and Gemini."""
        # Initialize embeddings and vector store
        self.embeddings = FastEmbedEmbeddings()
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
        )

        self.compressor = FlashrankRerank(model="ms-marco-TinyBERT-L-2-v2", top_n=5)

        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp", temperature=0.7, max_tokens=1000
        )

        # Conversation memory
        self.conversation_history: List[Any] = []

        # RAG prompt template
        self.rag_prompt = ChatPromptTemplate.from_template(
            """
You are a helpful AI assistant that answers questions based on the provided context and conversation history.

Context from documents:
{context}

Conversation History:
{history}

Current Question: {question}

Instructions:
1. Use the provided context to answer the question accurately
2. Consider the conversation history for context and continuity
3. If the answer is not in the provided context, say "I don't have enough information in the provided documents to answer that question."
4. Be conversational and helpful
5. Reference the source document when relevant

Answer:
"""
        )

    def get_available_files(self) -> List[str]:
        """Get list of available files in the vector store."""
        try:
            # Get all documents to extract unique filenames
            all_docs = self.vector_store.get()
            if all_docs and "metadatas" in all_docs:
                files = set()
                for metadata in all_docs["metadatas"]:
                    if metadata and "file" in metadata:
                        files.add(metadata["file"])
                return sorted(list(files))
            return []
        except Exception as e:
            print(f"Error getting available files: {e}")
            return []

    def search_documents(
        self, query: str, namespace: str = None, k: int = 10
    ) -> List[Document]:
        """Search for relevant documents in the vector store."""
        try:
            # if namespace:
            #     # Filter by file namespace
            #     filter_dict = {"file": namespace}
            #     results = self.vector_store.similarity_search_with_score(
            #         query, k=k, filter=filter_dict
            #     )
            # else:
            #     # Search across all documents
            #     results = self.vector_store.similarity_search_with_score(query, k=k)
            #     print(query, results)

            # # Return documents with relevance scores
            # documents = []
            # for doc, score in results:
            #     # Add relevance score to metadata
            #     doc.metadata["relevance_score"] = score
            #     documents.append(doc)

            if namespace:
                filter_dict = {"file": namespace}
                base_retriever = self.vector_store.as_retriever(
                    search_kwargs={"k": k, "filter": filter_dict}
                )
            else:
                base_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})

            # Wrap with compression retriever for reranking
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=self.compressor, base_retriever=base_retriever
            )

            # Get reranked documents
            documents = compression_retriever.invoke(query)

            # print(documents)
            # print(type(documents[0]))
            return documents
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []

    def format_conversation_history(self) -> str:
        """Format conversation history for the prompt."""
        if not self.conversation_history:
            return "No previous conversation."

        history_text = ""
        for i, message in enumerate(self.conversation_history[-6:]):  # Last 6 messages
            if isinstance(message, HumanMessage):
                history_text += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                history_text += f"Assistant: {message.content}\n"

        return history_text

    def format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents as context."""
        if not documents:
            return "No relevant documents found."

        context_parts = []
        for i, doc in enumerate(documents):
            file_name = doc.metadata.get("file", "Unknown")
            chunk_info = f"chunk {doc.metadata.get('chunk_index', '?')}/{doc.metadata.get('total_chunks', '?')}"
            relevance = doc.metadata.get("relevance_score", 0)

            context_parts.append(
                f"Document {i+1} (from {file_name}, {chunk_info}, relevance: {relevance:.3f}):\n"
                f"{doc.page_content}\n"
            )

        return "\n".join(context_parts)

    def ask_question(self, question: str, namespace: str = None) -> str:
        """Ask a question and get an answer using RAG."""
        try:
            # Search for relevant documents
            relevant_docs = self.search_documents(question, namespace)

            # Format context and history
            context = self.format_context(relevant_docs)
            history = self.format_conversation_history()

            # Generate response using Gemini
            prompt_input = {
                "context": context,
                "history": history,
                "question": question,
            }

            messages = self.rag_prompt.format_messages(**prompt_input)
            print(messages)
            response = self.llm.invoke(messages)

            # Add to conversation history
            self.conversation_history.append(HumanMessage(content=question))
            self.conversation_history.append(AIMessage(content=response.content))

            return response.content

        except Exception as e:
            error_msg = f"Error generating response: {e}"
            print(error_msg)
            return error_msg

    def clear_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("Conversation history cleared.")

    def show_conversation_stats(self):
        """Show conversation statistics."""
        print(f"Conversation length: {len(self.conversation_history)} messages")
        print(f"Available files: {', '.join(self.get_available_files())}")


def main():
    """Main interactive loop for the QnA bot."""
    print("====Initializing===")

    # Initialize bot
    bot = QnABot()

    # Show available files
    available_files = bot.get_available_files()
    if available_files:
        print(f"\nAvailable documents: {', '.join(available_files)}")
    else:
        print("\nNo documents found in the database. Please run fit_db.py first.")
        return

    print("\nCommands:")
    print("- Type your question to ask")
    print("- 'files' - show available files")
    print("- 'clear' - clear conversation history")
    print("- 'stats' - show conversation statistics")
    print("- 'quit' - exit the bot")
    print("- Use '@filename' at the start to search within a specific file")

    current_namespace = None

    while True:
        try:
            # Get user input
            user_input = input(f"\n[{current_namespace or 'All files'}] You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            elif user_input.lower() == "clear":
                bot.clear_conversation()
                continue
            elif user_input.lower() == "stats":
                bot.show_conversation_stats()
                continue
            elif user_input.lower() == "files":
                files = bot.get_available_files()
                print(f"Available files: {', '.join(files) if files else 'None'}")
                continue

            # Handle namespace selection
            if user_input.startswith("@"):
                parts = user_input[1:].split(" ", 1)
                if len(parts) == 2:
                    namespace, question = parts
                    if namespace in available_files:
                        current_namespace = namespace
                        user_input = question
                        print(f"Switched to namespace: {namespace}")
                    else:
                        print(
                            f"File '{namespace}' not found. Available files: {', '.join(available_files)}"
                        )
                        continue
                else:
                    # Just switching namespace
                    namespace = parts[0]
                    if namespace in available_files:
                        current_namespace = namespace
                        print(f"Switched to namespace: {namespace}")
                    elif namespace.lower() == "all":
                        current_namespace = None
                        print("Switched to search all files")
                    else:
                        print(
                            f"File '{namespace}' not found. Available files: {', '.join(available_files)}"
                        )
                    continue

            # Get answer from bot
            print("Bot: ", end="", flush=True)
            answer = bot.ask_question(user_input, current_namespace)
            print(answer)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
