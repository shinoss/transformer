import torch.nn as nn
import torch
import einops
import math
from utils import softmax
from typing import Callable, Optional, Tuple

def init_linear(weights, d_in, d_out):
    std = math.sqrt(2 / (d_in + d_out)) 
    min_init = -3.0 * std
    max_init = 3.0 * std
    torch.nn.init.trunc_normal_(weights, 0, std, min_init, max_init)

class Linear(nn.Module):
    def __init__(self, d_in, d_out, device=None, dtype=None):
        super().__init__()
        # NOTE: this is the row-major, non-transposed W 
        weights = torch.randn((d_out, d_in), device=device, dtype=dtype)
        self.w = nn.Parameter(weights)
        init_linear(self.w, d_in, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = einops.einsum(x, self.w, '... d_in, d_out d_in -> ... d_out')
        return out

class Embedding(nn.Module):
    def __init__(self, num_embeddings, d_model, device=None, dtype=None):
        super().__init__()
        self.embedding = nn.Parameter(
            torch.randn(num_embeddings, d_model, device=device, dtype=dtype)
        ) 
        torch.nn.init.trunc_normal_(self.embedding, 0, 1, -3, 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.scaler = 1.0 / d_model

    def forward(self, x):
        # x shape is (bsz, seq, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        square_sum = torch.sum(torch.square(x), dim=-1).unsqueeze(-1)
        rms = torch.sqrt(self.scaler * square_sum + self.eps)
        out = x / rms * self.gain
        return out.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = 8/3 * d_model
        # ensure d_ff is multiple of 64
        d_ff = math.ceil(d_ff / 64) * 64
        self.w1 = nn.Parameter(torch.randn(d_ff, d_model, device=device, dtype=dtype)) 
        self.w2 = nn.Parameter(torch.randn(d_model, d_ff, device=device, dtype=dtype)) 
        self.w3 = nn.Parameter(torch.randn(d_ff, d_model, device=device, dtype=dtype)) 

    def forward(self, x):
        w1_out = torch.matmul(x, self.w1.T)
        w1_out_silu = w1_out * torch.sigmoid(w1_out)
        w3_out = torch.matmul(x, self.w3.T)
        out = torch.matmul(w1_out_silu * w3_out, self.w2.T)
        return out


def rotate_half(x: torch.Tensor):
    # split emb dim into even and odd indices
    x_pairs = x.unflatten(-1, (-1, 2))
    x1 = x_pairs[...,0]
    x2 = x_pairs[...,1]
    x_rot = torch.stack((-x2,x1),-1).flatten(-2)
    return x_rot


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        theta_num = torch.arange(max_seq_len, device=device)
        k = torch.arange(1, d_k//2 + 1, device=device)
        theta_denom_exp = (2*k - 2)/d_k
        theta_denom = torch.tensor(theta) ** -theta_denom_exp
        freqs = torch.outer(theta_num, theta_denom)
        emb = torch.repeat_interleave(freqs, 2, -1)
        self.register_buffer('cos_computed',emb.cos(), persistent=False)
        self.register_buffer('sin_computed',emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_rot = rotate_half(x)
        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
        cos_range = self.cos_computed[token_positions]
        sin_range = self.sin_computed[token_positions]
        return x * cos_range + x_rot * sin_range


def sdpa(q, k, v, mask = None) -> torch.Tensor:
    d_k = k.shape[-1]
    qk = einops.einsum(q, k, '... sq d_k, ... sk d_k -> ... sq sk')
    pre_softmax = qk.div(math.sqrt(d_k))
    if mask is not None:
        pre_softmax = torch.where(
            mask,
            pre_softmax,
            float("-inf"),
        )
    out = torch.matmul(softmax(pre_softmax), v)
    return out 


class MHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: Optional[float] = None, max_seq_len: Optional[int] = None): 
        super().__init__()
        assert d_model % num_heads == 0
        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.qkv_proj = nn.Parameter(torch.randn(3 * d_model, d_model))
        self.o_proj = nn.Parameter(torch.randn(d_model, d_model))
        self.rope = None
        if theta is not None and max_seq_len is not None: 
            self.rope = RotaryPositionalEmbedding(theta, self.d_head, max_seq_len)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        *batch_dim, seq_len, _ = x.shape
        qkv = torch.matmul(x, self.qkv_proj.T)

        q, k, v = qkv.chunk(3, dim=-1)

        q = einops.rearrange(q, "... seq (h d) -> ... h seq d", h=self.num_heads)
        k = einops.rearrange(k, "... seq (h d) -> ... h seq d", h=self.num_heads)
        v = einops.rearrange(v, "... seq (h d) -> ... h seq d", h=self.num_heads)
        if self.rope is not None:
            q = self.rope(q, token_positions) 
            k = self.rope(k, token_positions)

        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=k.device).tril()

        attn_out = sdpa(q, k, v, mask) # (bsz, num_heads, seq_len, d_head)
        attn_out = einops.rearrange(
            attn_out, "... h seq d -> ... seq (h d)"
        ).contiguous()

        return torch.matmul(attn_out, self.o_proj.T)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_heads: int, theta: Optional[float] = None, max_seq_len: Optional[int] = None): 
        super().__init__()
        self.attn = MHA(d_model, num_heads, theta, max_seq_len)
        self.ln1 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor):
        y = x + self.attn(self.ln1(x))
        out = y + self.ffn(self.ln2(y))
        return out


class TransformerLM(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int, 
        d_ff: int, 
        num_heads: int,
        rope_theta: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, d_ff, num_heads, rope_theta, context_length)
            for _ in range(num_layers)
        ]) 
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.ln_final = RMSNorm(d_model)
        # No weight tying here
        self.lm_head = Linear(d_model, vocab_size)
        self.context_length = context_length

    def sample(self, probs, top_p):
        if top_p < 1.0:
            sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
            sorted_cumsum = torch.cumsum(sorted_probs, dim=-1)
            discard = (sorted_cumsum - sorted_probs) > top_p
            sorted_probs[discard] = 0
            # renormalize after filtering
            sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
            sampled_sorted = torch.multinomial(sorted_probs,1)
            sampled = torch.gather(indices, -1, sampled_sorted)
        else:
            sampled = torch.multinomial(probs,1)
        return sampled

    def generate(self, prompt: torch.Tensor, num_generate: int, temp: float, top_p: float, eos_token_id: int):
        generated = 0
        while generated < num_generate:
            logits = self.forward(prompt[:,-self.context_length:])
            if temp == 0.0:
                # greedy sampling
                next_token_ids = torch.argmax(logits[:,-1,:], dim=-1, keepdim=True)
            else:
                next_token_probs = softmax(logits[:,-1,:], -1, temp)
                next_token_ids = self.sample(next_token_probs, top_p)
            if (next_token_ids[:,-1] == eos_token_id).any().item():
                break;
            prompt = torch.concat((prompt, next_token_ids),dim=-1)
            generated += 1
        # only return the newly generated tokens
        return prompt[:, -num_generate:]

    def forward(self, x: torch.Tensor):
        embeddings = self.token_embeddings(x)
        for layer in self.layers:
            embeddings = layer(embeddings)
        normalized_embs = self.ln_final(embeddings)
        presoftmax_logits = self.lm_head(normalized_embs)
        return presoftmax_logits
