import gradio as gr

chat_history = [gr.ChatMessage(role="assistant", content=gr.Image("./img/mountains.jpg"))]

def info(message, history):
    return "Here is the media."

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(chat_history, type="messages")
    gr.ChatInterface(info, type="messages", chatbot=chatbot)

demo.launch()