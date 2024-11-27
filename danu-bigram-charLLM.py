import torch
import torch.nn as nn 
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

torch.manual_seed(1337) # generates the same random sequence

block_size = 8
batch_size = 32


data = open("data/shakespeare_char/input.txt").read()
vocab = sorted(list(set(data)))
vocab_size = len(vocab)

stoi = {ch:i for (i, ch) in enumerate(vocab)}
itoch = {i:ch for (i, ch) in enumerate(vocab)}
encode = lambda s: [stoi[ch] for ch in s] 
decode = lambda l: ''.join([itoch[i] for i in l])

n = int(0.9 * len(data))
train_data = torch.tensor(encode(data[:n]))
val_data = torch.tensor(encode(data[n:]))

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        # nn.Embedding provides one hot encoding. But here we use it as a probability table (current_word --> next_word) for the bigram LM.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # logis for the next word predictions. (B, T, C)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is a (B, T) array of indices in the current context. (B: blocks, T:time positions in each block)
        for _ in range(max_new_tokens): 
            logits, _ = self(idx)
            # We are taking the logits for the last time step
            logits = logits[:,-1,:] # becomes (B, C)
            probs = F.softmax(logits, dim=1) # Converting the values in the C dimension (1st dim) returning (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    pass


def train():
    # create a PyTorch optimiser
    m = BigramLanguageModel(vocab_size)
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for steps in tqdm(range(10000)):
        # sample a batch of data
        xb, yb = get_batch('train')
        # evaluate the loss
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print(loss.item())
    print(decode(m.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)[0].tolist()))



if __name__ == "__main__":
    train()
    
    