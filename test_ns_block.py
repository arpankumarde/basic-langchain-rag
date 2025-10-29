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


# Main application with sidebar navigation
with gr.Blocks(
    title="Namespace Chat Application",
    css="footer{display:none !important}",
) as demo:
    selected_namespace = gr.State(value="")
    message_count = gr.State(value=0)

    with gr.Row():
        # Sidebar
        with gr.Column(scale=1, min_width=250):
            gr.Markdown("### 📚 Namespaces")

            # Current namespace indicator
            current_ns_display = gr.Markdown("**Current:** None selected")

            gr.Markdown("---")
            gr.Markdown("### 🗂️ Available Namespaces")

            # Namespace selection buttons in sidebar
            namespace_buttons: list[tuple[gr.Button, str]] = []
            for ns in available_namespaces:
                btn = gr.Button(
                    ns.replace("_", " ").title(),
                    size="sm",
                    variant="secondary",
                    scale=1,
                )
                namespace_buttons.append((btn, ns))

            gr.Markdown("---")
            gr.Markdown("### 📊 Stats")
            stats_display = gr.Markdown("Messages: 0")

            gr.Markdown("---")
            gr.Markdown("### ⚙️ Actions")
            clear_btn_sidebar = gr.Button(
                "🗑️ Clear Chat", size="sm", variant="secondary"
            )
            reset_ns_btn = gr.Button(
                "🔄 Reset Namespace", size="sm", variant="secondary"
            )

        # Main content area
        with gr.Column(scale=4):
            # Namespace selector page
            with gr.Column(visible=True) as selector_page:
                gr.Markdown("# 🚀 Welcome to Namespace Chat")
                gr.Markdown("### Choose a namespace from the sidebar to start chatting")

                with gr.Row():
                    for i in range(0, len(available_namespaces), 3):
                        with gr.Column():
                            for j in range(i, min(i + 3, len(available_namespaces))):
                                ns = available_namespaces[j]
                                gr.Markdown(
                                    f"""
                                    <div style="padding: 20px; border: 2px solid #e0e0e0; border-radius: 8px; margin: 10px 0;">
                                        <h3>📁 {ns.replace('_', ' ').title()}</h3>
                                        <p style="color: #666;">Click in sidebar to select</p>
                                    </div>
                                    """
                                )

            # Chat page (initially hidden)
            with gr.Column(visible=False) as chat_page:
                namespace_header = gr.Markdown("# Chat Interface")

                chatbot = gr.Chatbot(type="messages", height=500, show_label=False)

                with gr.Row():
                    msg = gr.Textbox(
                        label="Your message",
                        placeholder="Ask your question...",
                        show_label=False,
                        scale=9,
                    )
                    send_btn = gr.Button("📤 Send", variant="primary", scale=1)

    # Functions for navigation
    def select_namespace(ns: str):
        return (
            gr.update(visible=False),  # Hide selector
            gr.update(visible=True),  # Show chat
            ns,  # Update selected namespace
            gr.update(
                value=f"# 💬 Chat - Namespace: **{ns.replace('_', ' ').title()}**"
            ),
            gr.update(value=f"**Current:** {ns.replace('_', ' ').title()}"),
            [],  # Clear chat history
            0,  # Reset message count
            gr.update(value="Messages: 0"),
        )

    def go_back():
        return (
            gr.update(visible=True),  # Show selector
            gr.update(visible=False),  # Hide chat
            "",  # Clear namespace
            gr.update(value="# Chat Interface"),
            gr.update(value="**Current:** None selected"),
            [],  # Clear chat history
            0,  # Reset message count
            gr.update(value="Messages: 0"),
        )

    def respond(message, history, namespace, msg_count):
        if not message.strip():
            return history, "", msg_count, gr.update(value=f"Messages: {msg_count}")
        bot_response = chat(message, history, namespace)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": bot_response})
        msg_count += 1
        return history, "", msg_count, gr.update(value=f"Messages: {msg_count}")

    def clear_chat():
        return [], 0, gr.update(value="Messages: 0")

    # Connect namespace selection buttons
    for btn, ns in namespace_buttons:
        btn.click(
            select_namespace,
            inputs=[gr.State(value=ns)],
            outputs=[
                selector_page,
                chat_page,
                selected_namespace,
                namespace_header,
                current_ns_display,
                chatbot,
                message_count,
                stats_display,
            ],
        )

    # Connect chat controls
    send_btn.click(
        respond,
        [msg, chatbot, selected_namespace, message_count],
        [chatbot, msg, message_count, stats_display],
    )
    msg.submit(
        respond,
        [msg, chatbot, selected_namespace, message_count],
        [chatbot, msg, message_count, stats_display],
    )

    # Sidebar actions
    clear_btn_sidebar.click(clear_chat, None, [chatbot, message_count, stats_display])
    reset_ns_btn.click(
        go_back,
        None,
        [
            selector_page,
            chat_page,
            selected_namespace,
            namespace_header,
            current_ns_display,
            chatbot,
            message_count,
            stats_display,
        ],
    )


demo.launch()
