import math
import argparse
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# --------------------- 1. 缩放点积注意力 ---------------------
def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
    d_k = q.size(-1)
    attn_scores = torch.matmul(q, k.swapaxes(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    # 因果掩码
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

    attn_weights = F.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output, attn_weights


# --------------------- 2. 多头注意力 ---------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int):
        super().__init__()
        assert hidden_size % num_attention_heads == 0, "hidden_size 必须能被 num_attention_heads 整除"
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.d_k = hidden_size // num_attention_heads

        self.w_q = nn.Linear(hidden_size, hidden_size)
        self.w_k = nn.Linear(hidden_size, hidden_size)
        self.w_v = nn.Linear(hidden_size, hidden_size)
        self.w_o = nn.Linear(hidden_size, hidden_size)

    def split_heads(self, x: torch.Tensor):
        batch_size, seq_len, hidden_size = x.size()
        return x.view(batch_size, seq_len, self.num_attention_heads, self.d_k).swapaxes(1, 2)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        attn_output, attn_weights = scaled_dot_product_attention(q, k, v, mask)

        attn_output = attn_output.swapaxes(1, 2).reshape(batch_size, -1, self.hidden_size)
        output = self.w_o(attn_output)
        return output, attn_weights


# --------------------- 3. 前馈网络 ---------------------
class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, d_ff: int, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, d_ff)
        self.linear2 = nn.Linear(d_ff, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


# --------------------- 4. 完整 Transformer Encoder 层 ---------------------
class TransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, d_ff: int, dropout=0.2):
        super().__init__()
        self.mha = MultiHeadAttention(hidden_size, num_attention_heads)
        self.ffn = FeedForward(hidden_size, d_ff, dropout)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.mha(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x


# --------------------- 位置编码 ---------------------
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=2048, batch_first=True):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)) #频率系数
        if batch_first:
            pe = torch.zeros(1, max_len, embed_dim)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            pe = torch.zeros(max_len, 1, embed_dim)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        if self.batch_first:
            x = x + self.pe[:, :x.size(1)]
        else:
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# --------------------- 因果掩码 ---------------------
def generate_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
    return mask


# --------------------- 数据模块 ---------------------
def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text):
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


# ---------------------手写的模块构建的语言模型 ---------------------
class TransformerLM(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, num_heads, hidden_dim, num_layers, dropout=0.3
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)


        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        device = x.device

        x = self.embedding(x) * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)

        # 因果掩码
        mask = generate_causal_mask(seq_len, device)


        for layer in self.layers:
            x = layer(x, mask)

        logits = self.fc(x)
        return logits


# --------------------- 训练 / 评估 ---------------------
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = total_tokens = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)


# --------------------- 4 种采样策略 ---------------------
def generate_text(
    model, char2idx, idx2char,
    start_text="今天",
    length=200,
    temperature=0.7,
    strategy="top_p",
    top_k=50,
    top_p=0.9,
    beam_width=5,
    device="cpu"
):
    model.eval()
    input_ids = [char2idx[c] for c in start_text if c in char2idx]

    if strategy == "greedy":
        for _ in range(length):
            x = torch.tensor([input_ids], device=device)
            logits = model(x)[:, -1, :] / temperature
            next_id = logits.argmax(-1).item()
            input_ids.append(next_id)

    elif strategy == "beam":
        sequences = [(input_ids.copy(), 0.0)]
        for _ in range(length):
            candidates = []
            for seq, score in sequences:
                x = torch.tensor([seq], device=device)
                with torch.no_grad():
                    logits = model(x)[:, -1, :] / temperature
                log_probs = F.log_softmax(logits, dim=-1)
                top_v, top_i = log_probs.topk(beam_width)
                for v, i in zip(top_v[0], top_i[0]):
                    new_seq = seq + [i.item()]
                    new_score = score + v.item()
                    candidates.append((new_seq, new_score))
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            sequences = candidates
        input_ids = sequences[0][0]

    elif strategy == "top_k":
        for _ in range(length):
            x = torch.tensor([input_ids], device=device)
            logits = model(x)[:, -1, :] / temperature
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            input_ids.append(next_id)

    elif strategy == "top_p":
        for _ in range(length):
            x = torch.tensor([input_ids], device=device)
            logits = model(x)[:, -1, :] / temperature
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            input_ids.append(next_id)

    return "".join([idx2char[i] for i in input_ids])


# --------------------- 主函数 ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--corpus", default="*.txt")
    parser.add_argument("--save",       default="transformer.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | 模型：完全手写 Transformer")

    text = load_corpus(args.corpus)
    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds = CharDataset(val_text, char2idx, args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = TransformerLM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>5} | {'Train Loss':>10} | {'Train PPL':>8} | {'Val Loss':>10} | {'Val PPL':>8}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx, "args": vars(args),
                "idx2char": idx2char
            }, args.save)

        print(f"{epoch:>5} | {tr_loss:>10.4f} | {tr_ppl:>8.2f} | {va_loss:>10.4f} | {va_ppl:>8.2f}{marker}")

    print(f"\n训练完成！最佳 PPL: {best_val_ppl:.2f}")

if __name__ == "__main__":
    main()