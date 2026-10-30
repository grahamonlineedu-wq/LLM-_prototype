import gradio as gr
from src.model import LLM # your model class
import torch

# load your checkpoint
model = LLM.from_pretrained("checkpoints/best.pt")
model.eval()

def generate(prompt):
    return model.generate(prompt, max_tokens=100)

gr.Interface(fn=generate, inputs="text", outputs="text", title="LLM Prototype 6.8M").launch()
