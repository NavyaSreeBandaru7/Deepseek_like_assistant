import gradio as gr
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

# Configuration - Using a smaller model that fits in GPU memory
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"  # Smaller 1.3B parameter model
OFFLOAD_DIR = "./offload"  # Directory for offloaded weights
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 1024  # Reduced from 2048
TEMPERATURE = 0.7

# Create offload directory if needed
os.makedirs(OFFLOAD_DIR, exist_ok=True)

# Load model with offloading support
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Check if we have a GPU and use appropriate settings
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            offload_folder=OFFLOAD_DIR,
            offload_state_dict=True
        )
    else:
        # CPU-only configuration
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map=None,
            offload_folder=OFFLOAD_DIR
        )
        model = model.to(DEVICE)
        
    print(f"✅ Loaded {MODEL_NAME} on {DEVICE}")
except Exception as e:
    print(f"❌ Model loading failed: {str(e)}")
    raise RuntimeError("Model initialization error") from e

def generate_response(
    message: str,
    history: list[tuple[str, str]],
    system_prompt: str = "You are a helpful AI assistant",
    temperature: float = TEMPERATURE,
    max_new_tokens: int = MAX_NEW_TOKENS
):
    """Generation with streaming and memory management"""
    try:
        # Build conversation context
        messages = [{"role": "system", "content": system_prompt}]
        for user, assistant in history:
            messages.extend([
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant}
            ])
        messages.append({"role": "user", "content": message})

        # Tokenize with chat template
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        # Streamer setup for real-time output
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            timeout=20.0,
            skip_special_tokens=True
        )

        # Asynchronous generation
        generation_kwargs = dict(
            inputs=inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # Stream output incrementally
        partial_message = ""
        for new_token in streamer:
            partial_message += new_token
            yield partial_message

    except torch.cuda.OutOfMemoryError:
        yield "⚠️ Error: GPU memory exhausted. Please simplify your query."
    except Exception as e:
        yield f"🔧 System error: {str(e)}"

# Create a custom chat interface without unsupported parameters
def create_chat_interface():
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox(
            label="Ask DeepSite anything...",
            placeholder="Type your message here...",
        )
        clear = gr.Button("Clear")
        
        system_prompt = gr.Textbox(
            "You are an expert AI assistant. Provide detailed, helpful answers.",
            label="System Role"
        )
        temperature = gr.Slider(0.1, 1.0, TEMPERATURE, label="Creativity")
        max_tokens = gr.Slider(128, 2048, MAX_NEW_TOKENS, label="Max Response Length")

        def respond(message, chat_history, system_prompt, temperature, max_tokens):
            bot_message = ""
            for response in generate_response(message, chat_history, system_prompt, temperature, max_tokens):
                bot_message = response
            chat_history.append((message, bot_message))
            return "", chat_history

        msg.submit(
            respond,
            [msg, chatbot, system_prompt, temperature, max_tokens],
            [msg, chatbot]
        )
        clear.click(lambda: None, None, chatbot, queue=False)
        
    return demo

# Production UI with Gradio
with gr.Blocks(
    title="DeepSeek-like Chat",
    theme=gr.themes.Soft(spacing_size="md", font=[gr.themes.GoogleFont("Open Sans")]),
    css="footer {visibility: hidden} .gradio-container {max-width: 700px !important}"
) as interface:
    gr.Markdown("# 🧠 DeepSeek-like Assistant")
    
    with gr.Row():
        with gr.Column(scale=3):
            system_prompt = gr.Textbox(
                "You are an expert AI assistant. Provide detailed, helpful answers.",
                label="System Role"
            )
            temperature = gr.Slider(0.1, 1.0, TEMPERATURE, label="Creativity")
            max_tokens = gr.Slider(128, 2048, MAX_NEW_TOKENS, label="Max Response Length")
        
        with gr.Column(scale=7):
            # Use our custom chat interface
            chatbot = gr.Chatbot()
            msg = gr.Textbox(
                label="Ask DeepSite anything...",
                placeholder="Type your message here...",
            )
            clear = gr.Button("Clear")
            
            examples = gr.Examples(
                examples=[
                    "Explain quantum computing in simple terms",
                    "How do I implement a Python REST API?",
                    "Write a haiku about AI ethics"
                ],
                inputs=msg
            )
    
    gr.Markdown("> Optimized for GPU Memory | Built with Hugging Face Transformers")
    
    # Define the response function
    def respond(message, chat_history, system_prompt, temperature, max_tokens):
        bot_message = ""
        for response in generate_response(message, chat_history, system_prompt, temperature, max_tokens):
            bot_message = response
        chat_history.append((message, bot_message))
        return "", chat_history

    # Set up the interaction
    msg.submit(
        respond,
        [msg, chatbot, system_prompt, temperature, max_tokens],
        [msg, chatbot]
    )
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    interface.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
