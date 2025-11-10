import os
import getpass
from dotenv import load_dotenv
import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

# Initialize Gemini chat model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp", temperature=0.2, max_tokens=512
)


def respond(user_message: str, chat_history: list[tuple[str, str]]):
    """Call Gemini and return updated chat history and cleared input."""
    if not user_message or not user_message.strip():
        return chat_history, ""

    # Build messages for the model
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_message},
    ]

    try:
        # Invoke the model
        response = llm.invoke(messages)

        # Try multiple ways to extract text depending on return shape
        if hasattr(response, "content"):
            bot_text = response.content
        elif isinstance(response, dict) and "content" in response:
            bot_text = response["content"]
        else:
            # Fallback to string representation
            bot_text = str(response)

    except Exception as e:
        bot_text = f"Error calling Gemini: {e}"

    # Update chat history used by gr.Chatbot (list of (user, bot) tuples)
    chat_history = chat_history or []
    chat_history.append((user_message, bot_text))

    return chat_history, ""


with gr.Blocks(title="Gemini Chat", css="footer{display:none !important}") as demo:
    gr.Markdown("# Gemini Chat")
    chatbot = gr.Chatbot(elem_id="chatbot", height=450, type="messages")
    with gr.Row():
        txt = gr.Textbox(placeholder="Ask something...", show_label=False)
        send = gr.Button("Send", variant="primary")

    # Wire up interactions
    send.click(respond, [txt, chatbot], [chatbot, txt])
    txt.submit(respond, [txt, chatbot], [chatbot, txt])

demo.launch()
