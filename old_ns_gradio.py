import gradio as gr
import os


def get_namespaces(directory: str):
    namespaces: list[str] = []
    if os.path.exists(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                namespaces.append(item.lower())
    return namespaces


def chat(message, history, namespace):
    # Dummy response that includes the namespace
    return f"[Namespace: {namespace}] You said: {message}"


# Get available namespaces
data_directory = "./data_ns"
available_namespaces = get_namespaces(data_directory)

if not available_namespaces:
    available_namespaces = ["default", "example1", "example2"]  # Fallback for demo


# Create the namespace selection interface
def create_namespace_selector():
    with gr.Blocks() as namespace_page:
        gr.Markdown("# Select a Namespace")
        gr.Markdown("Choose a namespace to start chatting:")

        with gr.Row():
            for ns in available_namespaces:
                gr.Button(ns, size="lg", variant="primary")

        selected_namespace = gr.State(value="")

    return namespace_page, selected_namespace


# Create the chat interface
def create_chat_interface(namespace: str):
    with gr.Blocks() as chat_page:
        gr.Markdown(f"# Chat Interface - Namespace: **{namespace}**")

        chatbot = gr.Chatbot(type="messages", height=500)
        msg = gr.Textbox(
            label="Your message",
            placeholder=f"Ask questions about {namespace}...",
            show_label=False,
        )

        with gr.Row():
            submit = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear")
            back = gr.Button("Change Namespace")

        def respond(message, history):
            if not message.strip():
                return history
            bot_response = chat(message, history, namespace)
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": bot_response})
            return history

        def clear_chat():
            return []

        submit.click(respond, [msg, chatbot], [chatbot]).then(lambda: "", None, [msg])
        msg.submit(respond, [msg, chatbot], [chatbot]).then(lambda: "", None, [msg])
        clear.click(clear_chat, None, [chatbot])

    return chat_page, back


# Main application with navigation
with gr.Blocks(
    title="Namespace Chat Application",
    css="footer{display:none !important}",
    theme=gr.themes.Soft(),
) as demo:
    current_page = gr.State(value="selector")
    selected_namespace = gr.State(value="")

    # Namespace selector page
    with gr.Column(visible=True) as selector_page:
        gr.Markdown("# Select a Namespace")
        gr.Markdown("Choose a namespace to start chatting:")

        namespace_buttons: list[tuple[gr.Button, str]] = []
        with gr.Row():
            for ns in available_namespaces:
                btn = gr.Button(
                    ns.replace("_", " ").title(), size="lg", variant="primary", scale=1
                )
                namespace_buttons.append((btn, ns))

    # Chat page (initially hidden)
    with gr.Column(visible=False) as chat_page:
        namespace_display = gr.Markdown("# Chat Interface")

        chatbot = gr.Chatbot(type="messages", height=500)
        msg = gr.Textbox(
            label="Your message", placeholder="Ask your question...", show_label=False
        )

        with gr.Row():
            submit = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear")
            back = gr.Button("Change Namespace")

    # Functions for navigation
    def select_namespace(ns: str):
        return (
            gr.update(visible=False),  # Hide selector
            gr.update(visible=True),  # Show chat
            ns,  # Update selected namespace
            gr.update(
                value=f"# Chat Interface - Namespace: **{ns}**"
            ),  # Update display
            [],  # Clear chat history
        )

    def go_back():
        return (
            gr.update(visible=True),  # Show selector
            gr.update(visible=False),  # Hide chat
            "",  # Clear namespace
            gr.update(value="# Chat Interface"),  # Reset display
            [],  # Clear chat history
        )

    def respond(message, history, namespace):
        if not message.strip():
            return history, ""
        bot_response = chat(message, history, namespace)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": bot_response})
        return history, ""

    def clear_chat():
        return []

    # Connect namespace selection buttons
    for btn, ns in namespace_buttons:
        btn.click(
            select_namespace,
            inputs=[gr.State(value=ns)],
            outputs=[
                selector_page,
                chat_page,
                selected_namespace,
                namespace_display,
                chatbot,
            ],
        )

    # Connect chat controls
    submit.click(respond, [msg, chatbot, selected_namespace], [chatbot, msg])
    msg.submit(respond, [msg, chatbot, selected_namespace], [chatbot, msg])
    clear.click(clear_chat, None, [chatbot])
    back.click(
        go_back,
        None,
        [selector_page, chat_page, selected_namespace, namespace_display, chatbot],
    )


demo.launch()
