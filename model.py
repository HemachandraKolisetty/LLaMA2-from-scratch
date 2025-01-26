import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # Number of heads for Q
    n_kv_heads: Optional[int] = None # Number of heads for K and V
    vocab_size: int = -1 # depends on the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Neede for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

def precompute_theta_pos_frequencies(head_dim: int, seq_len:int, device:str, theta: float = 10000.0):
    # As written in the paper, the dim must be even
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"

    # Build the theta parameters
    # According to the formula theta_i = 10000 ^ (-2(i - 1)/dim) for i = [1, 2, ....., dim / 2]
    # Shape: (Head_dim / 2)
    theta_number = torch.arange(0, head_dim, 2).float()
    # Shape: (Head_dim / 2)
    theta = 1.0 / (theta ** (theta_number / head_dim)).to(device)
    # Construct the positions (the "m" parameter)
    # Shape: (seq_len)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position using the outer product
    # Shape: (seq_len) outer_product* (head_dim / 2) -> (seq_len, head_dim / 2)
    freqs = torch.outer(m, theta).float()
    # We can compute the complex numbers in the polar form c = R * exp(i * m * theta), where R = 1 as follows:
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embedding(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # (B, Seq_len, H, Head_dim) -> (B, Seq_len, H, Head_dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (seq_len, head_dim / 2) -> (1, seq_len, 1, head_dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (B, seq_len, H, head_dim/2) * (1, seq_len, 1, head_dim/2) -> (B, seq_len, H, head_dim/2)
    x_rotated = x_complex * freqs_complex # element-wise multiplication
    # (B, seq_len, H, head_dim / 2) -> (B, seq_len, H, head_dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, seq_len, H, head_dim / 2, 2) -> (B, seq_len, H, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # gamma
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor):
        # (B, Seq_len, dim) * (B, Seq_len, 1) -> (B, seq_len, dim)
        # rsqrt(x) -> 1 / sqrt()x
        return x * torch.rsqrt((x ** 2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (B, seq_len, dim) * (dim) -> (B, seq_len, dim)
        return self._norm(x.float()).type_as(x) * self.weight

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    # extract the shape
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )

class SelfAttention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads_q = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // self.n_heads_q

        self.wq = nn.Linear(args.dim, self.n_heads_q * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # extract the dimensions
        batch_size, seq_len, _ = x.shape # (B, 1, dim) , seq_len is 1 during inference

        # compute query: (B, Seq_len, dim) -> (B, Seq_len, n_heads * head_dim)
        xq = self.wq(x)
        # (B, seq_len, dim) -> (B, Seq_len, n_kv_heads * head_dim)
        xk = self.wk(x)
        xv = self.wv(x)
        
        # (B, seq_len, n_heads_q * head_dim) -> (B, seq_len, n_heads_q, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, seq_len, n_kv_heads * head_dim) -> (B, seq_len, n_kv_heads, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Apply rotary positional encodings
        xq = apply_rotary_embedding(xq, freqs_complex, x.device)
        xk = apply_rotary_embedding(xk, freqs_complex, x.device)

        # replace the entry in the cache for this token
        self.cache_k[:batch_size, start_pos:start_pos+1] = xk
        self.cache_v[:batch_size, start_pos:start_pos+1] = xv

        # extract all the keys and values of all previous tokens
        # (B, seq_len_kv, n_kv_heads, head_dim)
        keys = self.cache_k[:batch_size, :start_pos + 1]
        values = self.cache_v[:batch_size, :start_pos + 1]

        # Repeat the heads of keys and values to match the heads of the queries
        # (B, seq_len_kv, n_kv_heads, head_dim) -> (B, seq_len_kv, n_heads_q, head_dim)
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # (B, 1, H_Q, head_dim) -> (B, H_Q, 1, head_dim)
        xq = xq.transpose(1, 2)
        # (B, seq_len_kv, H_q, head_dim) -> (B, H_q, seq_len_kv, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (B, H_Q, 1, seq_len_kv)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, H_Q, 1, head_dim)
        ouput = torch.matmul(scores, values)

        # Finally we concatenate the ouputs of all the attention heads
        # (B, H_Q, 1, head_dim) -> (B, 1, H_Q, head_dim) -> (B, 1, dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output) # (B, 1, dim)

class FeedForward(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # round the hidden dim to the nearest multiple of the multiple_of parameter
        hidden_dim =  args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias = False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias = False)
    
    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)

        return x

class EncoderBlock(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # Normalization before self attention
        self.attention_norm = RMSNorm(self.dim, eps=args.norm_eps)
        # Normalization before feed forward
        self.ffn_norm = RMSNorm(self.dim, eps=args.norm_eps)
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, seq_len, dim) + (B, seq_len, dim) -> (B, seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex) # TODO: verify if this implementation is right. In the lecture self.attention.forward() method is called instead
        out = h + self.feed_forward.forward(self.ffn_norm(x))
        return out

class Transformer(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1 , 'Vocab size must be initialized'

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)
        
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(args.dim // args.n_heads, args.max_seq_len * 2, device = args.device)
    
    def forward(self, token: torch.Tensor, start_pos: int):
        # (B, Seq_len)
        batch_size, seq_len = token.shape
        assert seq_len == 1, 'Only one token at a time can be processed during inference. (Using KV Cache)'

        # (B, Seq_len) -> (B, Seq_len, dim)
        h = self.tok_embeddings(token)

        # Retreive the paid (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)

        h = self.norm(h)
        output = self.output(h).float()
        return output
