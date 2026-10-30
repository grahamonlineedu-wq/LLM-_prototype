import torch, tiktoken, os, requests
from main import TinyLLM, Config

enc = tiktoken.get_encoding("gpt2")
config = Config()
config.block_size = 128

# Download tiny shakespeare (~1MB, perfect for phone CPU)
if not os.path.exists("input.txt"):
    print("Downloading data...")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data = requests.get(url).text
    open("input.txt","w",encoding="utf-8").write(data)

text = open("input.txt", encoding="utf-8").read()
print(f"Dataset {len(text)} chars")
data_ids = enc.encode(text)
print(f"{len(data_ids)} tokens")

# Train/val split
n = int(0.9 * len(data_ids))
train_data = torch.tensor(data_ids[:n], dtype=torch.long)
val_data = torch.tensor(data_ids[n:], dtype=torch.long)

def get_batch(split, batch_size=4):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+config.block_size] for i in ix])
    return x, y

model = TinyLLM(config)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
print(f"Training {sum(p.numel() for p in model.parameters())/1e6:.2f}M params on CPU...")

for step in range(300): # increase to 2000 later
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 20 == 0:
        # val loss
        model.eval()
        with torch.no_grad():
            xv, yv = get_batch('val')
            _, vloss = model(xv, yv)
        model.train()
        print(f"step {step} train {loss.item():.3f} val {vloss.item():.3f}")

        # sample
        prompt = "Hello"
        ids = torch.tensor([enc.encode(prompt)], dtype=torch.long)
        out = model.generate(ids, max_new_tokens=60)
        print("SAMPLE:", enc.decode(out[0].tolist())[:150], "\n")

torch.save(model.state_dict(), "tinyllm.pt")
print("Saved tinyllm.pt")
