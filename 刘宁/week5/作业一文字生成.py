
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
import argparse


# --------------------- 模型结构 ---------------------
def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
    d_k = q.size(-1)
    attn_scores = torch.matmul(q, k.swapaxes(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
    attn_weights = F.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output, attn_weights

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int):
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.d_k = hidden_size // num_attention_heads
        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)
        self.w_o = nn.Linear(hidden_size, hidden_size)
    def split_heads(self, x):
        B, L, D = x.shape
        return x.view(B, L, self.num_attention_heads, self.d_k).swapaxes(1,2)
    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        q = self.split_heads(self.w_q(q))
        k = self.split_heads(self.w_k(k))
        v = self.split_heads(self.w_v(v))
        attn_output, _ = scaled_dot_product_attention(q, k, v, mask)
        attn_output = attn_output.swapaxes(1,2).reshape(B, -1, self.hidden_size)
        return self.w_o(attn_output), None

class FeedForward(nn.Module):
    def __init__(self, hidden_size, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, d_ff)
        self.linear2 = nn.Linear(d_ff, hidden_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, d_ff, dropout=0.2):
        super().__init__()
        self.mha = MultiHeadAttention(hidden_size, num_heads)
        self.ffn = FeedForward(hidden_size, d_ff, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        attn_out, _ = self.mha(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=2048, batch_first=True):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        if batch_first:
            pe = torch.zeros(1, max_len, embed_dim)
            pe[0,:,0::2] = torch.sin(position * div_term)
            pe[0,:,1::2] = torch.cos(position * div_term)
        else:
            pe = torch.zeros(max_len, 1, embed_dim)
            pe[:,0,0::2] = torch.sin(position * div_term)
            pe[:,0,1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        if self.batch_first:
            x = x + self.pe[:, :x.size(1)]
        else:
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def generate_causal_mask(seq_len, device):
    return torch.tril(torch.ones((seq_len, seq_len), device=device))

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, hidden_dim, dropout) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(embed_dim, vocab_size)
    def forward(self, x):
        L = x.size(1)
        x = self.embedding(x) * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)
        mask = generate_causal_mask(L, x.device)
        for layer in self.layers:
            x = layer(x, mask)
        return self.fc(x)

# --------------------- 采样策略 ---------------------
def generate_text(
    model, char2idx, idx2char,
    start_text="今天", length=200, temperature=0.7,
    strategy="top_p", top_k=50, top_p=0.9, beam_width=5, device="cpu"
):
    model.eval()
    input_ids = [char2idx[c] for c in start_text if c in char2idx]

    if strategy == "greedy":
        for _ in range(length):
            x = torch.tensor([input_ids], device=device)
            logits = model(x)[:, -1] / temperature
            next_id = logits.argmax().item()
            input_ids.append(next_id)

    elif strategy == "beam":
        sequences = [(input_ids.copy(), 0.0)]
        for _ in range(length):
            candidates = []
            for seq, score in sequences:
                x = torch.tensor([seq], device=device)
                with torch.no_grad():
                    logits = model(x)[:, -1] / temperature
                lp = F.log_softmax(logits, dim=-1)
                v, i = lp.topk(beam_width)
                for vv, ii in zip(v[0], i[0]):
                    candidates.append((seq + [ii.item()], score + vv.item()))
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            sequences = candidates
        input_ids = sequences[0][0]

    elif strategy == "top_k":
        for _ in range(length):
            x = torch.tensor([input_ids], device=device)
            logits = model(x)[:, -1] / temperature
            k_val = torch.topk(logits, top_k)[0][:, -1, None]
            logits[logits < k_val] = -float('inf')
            p = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(p, 1).item()
            input_ids.append(next_id)

    elif strategy == "top_p":
        for _ in range(length):
            x = torch.tensor([input_ids], device=device)
            logits = model(x)[:, -1] / temperature
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_p = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            drop = cum_p > top_p
            drop[..., 1:] = drop[..., :-1].clone()
            drop[..., 0] = False
            logits[:, sorted_idx[drop]] = -float('inf')
            p = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(p, 1).item()
            input_ids.append(next_id)

    return "".join([idx2char[i] for i in input_ids])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("transformer.pt", map_location=device)

    char2idx = checkpoint["char2idx"]
    idx2char = checkpoint["idx2char"]
    args = argparse.Namespace(**checkpoint["args"])

    model = TransformerLM(
        vocab_size=len(char2idx),
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("==== 独立文本生成（输入 q 退出）====")
    while True:
        prompt = input("\n开头：")
        if prompt in ["q"]:
            break
        res = generate_text(
            model=model, char2idx=char2idx, idx2char=idx2char,
            start_text=prompt, length=200,
            strategy="top_p",  # 随便换
            device=device
        )
        print("生成：", res)

if __name__ == "__main__":
    main()