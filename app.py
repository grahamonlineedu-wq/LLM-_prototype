import gradio as gr
import torch
import tiktoken
from src.model import TinyLLM, Config

# Setup
config = Config()
config.block_size = 128
enc = tiktoken.get_encoding("gpt2")

# Load model
model = TinyLLM(config)

# Try both checkpoint locations
import os
ckpt_paths = ["checkpoints/tinyllm.pt", "tinyllm.pt", "checkpoints/best.pt"]
ckpt = None
for p in ckpt_paths:
    if os.path.exists(p):
        ckpt = p
        break

if ckpt is None:
    print(f"No checkpoint found, tried: {ckpt_paths}")
    print("Training one first or downloading from HF...")
    from huggingface_hub import hf_hub_download
    ckpt = hf_hub_download("grahmseinstein/tinyllm-28m", "tinyllm.pt", local_dir=".")

print(f"Loading checkpoint: {ckpt}")
model.load_state_dict(torch.load(ckpt, map_location='cpu'))
model.eval()

print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

def generate(prompt, max_tokens):
    if not prompt or prompt.strip() == "":
        prompt = "Hello"
    # Encode
    ids = torch.tensor([enc.encode(prompt)], dtype=torch.long)
    # Generate
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=int(max_tokens))
    # Decode
    text = enc.decode(out[0].tolist())
    return text

# Gradio UI
with gr.Blocks(title="TinyLLM 28M - grahmseinstein") as demo:
    gr.Markdown("# TinyLLM 28M Prototype\nby grahmseinstein - GPT2 50257 tokens")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt", value="Hello, I am a", lines=2)
        max_tokens = gr.Slider(20, 200, value=100, step=10, label="Max tokens")
    output = gr.Textbox(label="Generated", lines=10)
    btn = gr.Button("Generate")
    btn.click(fn=generate, inputs=[prompt, max_tokens], outputs=output)
    prompt.submit(fn=generate, inputs=[prompt, max_tokens], outputs=output)

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
