<<<<<<< HEAD
#llm-prototype
=======
# TinyLLM - Phone CPU Prototype

Built on phone (Termux + Ubuntu proot), Torch CPU 2.15.0
Status: CPU stage complete, val loss 5.67 @ 300 steps

## Architecture
- GPT2 tokenizer (50257 vocab)
- 4 layers, 4 heads, 256 embd ~6.8M params
- Manual CausalSelfAttention (fixed Torch 2.15 is_causal bug)

## Files
- main.py - model definition + overfit test
- train.py - train on tinyshakespeare
- train2.py - resume to 2000 steps

## Results
step 280 train 5.765 val 5.670

## Run
python3 main.py
python3 train.py

## Next: GPU stage -> n_layer=12, batch 32, cuda
>>>>>>> 5ac3a79 (feat: CPU prototype working - torch 2.15 cpu, 6.8M model, val 5.67 @ 280 steps)
