import torch, tiktoken
from main import TinyLLM, Config
import os

enc = tiktoken.get_encoding("gpt2")
config = Config()
config.block_size = 128

text = open("input.txt", encoding="utf-8").read()
data_ids = torch.tensor(enc.encode(text), dtype=torch.long)
n = int(0.9 * len(data_ids))
train_data, val_data = data_ids[:n], data_ids[n:]

def get_batch(split, bs=8):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - config.block_size, (bs,))
    x = torch.stack([d[i:i+config.block_size] for i in ix])
    y = torch.stack([d[i+1:i+1+config.block_size] for i in ix])
    return x, y

model = TinyLLM(config)
model.load_state_dict(torch.load("tinyllm.pt", map_location="cpu"))
opt = torch.optim.AdamW(model.parameters(), lr=1e-4) # lower lr for resume

for step in range(300, 2000):
    xb, yb = get_batch('train', bs=8)
    _, loss = model(xb, yb)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0:
        model.eval()
        with torch.no_grad():
            xv, yv = get_batch('val', bs=8)
            _, vloss = model(xv, yv)
        model.train()
        print(f"step {step} train {loss.item():.3f} val {vloss.item():.3f}")
        if step % 500 == 0:
            torch.save(model.state_dict(), f"tinyllm_{step}.pt")

torch.save(model.state_dict(), "tinyllm_final.pt")
print("DONE final")
