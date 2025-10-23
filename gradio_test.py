import gradio as gr


def chat(message, history):
    # Dummy response
    return f"You said: {message}"


demo = gr.ChatInterface(
    fn=chat,
    title="Basic Chat Application",
    description="A simple chat interface (dummy responses)",
    analytics_enabled=False,
)

demo.launch()
