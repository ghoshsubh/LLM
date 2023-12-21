import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional
import json
import numpy as np
import sys
from utils import *


def Compute_angles(head_dim:int, seq_len:int, device:str, theta:int = 10000)->torch.tensor:

  theta_numerator = torch.arange(0, head_dim, 2).float(); '(hs // 2, )'
  theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device); '(hs // 2, )'
  m = torch.arange(seq_len).to(device); "(T, )"
  angles = torch.outer(m, theta); '(T, hs // 2)'
  return angles.to(torch.float16)

def apply_rotation(x:torch.tensor, angles:torch.tensor)->torch.tensor:
  B, T, nh, hs = x.shape
  x_new = x.reshape(B, T, nh, hs // 2, 2); '(B, T, nh, hs) ----> (B, T, nh, hs // 2, 2)'

  angles = angles.unsqueeze(0).unsqueeze(2); '(T, hs // 2) ---> (1, T, 1, hs // 2)'
  angles = angles.repeat(B, 1, nh, 1); '(1, T, 1, hs // 2) ---> (B, T, nh, hs // 2)'

  cos_angles = torch.cos(angles).unsqueeze(4).unsqueeze(5); '(B, T, nh, hs // 2) ---> (B, T, nh, hs // 2, 1, 1)'
  sin_angles = torch.sin(angles).unsqueeze(4).unsqueeze(5); '(B, T, nh, hs // 2) ---> (B, T, nh, hs // 2, 1, 1)'


  new_cs1 = torch.cat((cos_angles, -1*sin_angles), dim = -1); '(B, T, nh, hs // 2, 1, 1) last dim concat (B, T, nh, hs // 2, 1, 1) ---> (B, T, nh, hs // 2, 1, 2)'
  new_cs2 = torch.cat((sin_angles, cos_angles), dim = -1); '(B, T, nh, hs // 2, 1, 1) last dim concat (B, T, nh, hs // 2, 1, 1) ---> (B, T, nh, hs // 2, 1, 2)'

  """ we make a 2*2 matrix for each angle t, such that the matrix is [[cost, -sint],
                                                                    [sint, cost]]
  """
  new_angles = torch.cat((new_cs1, new_cs2), dim = -2); '(B, T, nh, hs // 2, 1, 2) last dim concat (B, T, nh, hs // 2, 1, 2) ---> (B, T, nh, hs // 2, 2, 2)'

  rotated_x = new_angles @ x_new.unsqueeze(5); '(B, T, nh, hs // 2, 2, 2) @ (B, T, nh, hs // 2, 2, 1) ----> (B, T, nh, hs // 2, 2, 1)'

  rotated_x1 = rotated_x.squeeze(); '(B, T, nh, hs // 2, 2, 1) ---> (B, T, nh, hs // 2, 2)'

  rotated_x2 = rotated_x1.reshape(B, T, nh, hs); '(B, T, nh, hs // 2, 2) ---> (B, T, nh, hs)'

  return rotated_x2

class RMSNorm(nn.Module):
  def __init__(self, args)->None:
    super().__init__()
    self.eps = args.norm_eps
    self.weight = nn.Parameter(torch.ones(args.dim)).to(dtype = torch.float16)

  def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)

  def forward(self, x):
    return self.weight.to(x.device) * self._norm(x.float()).type_as(x); '(C, ) * (B, T, C) ---> (B, T, C)' 

class GroupQueryAtention(nn.Module):
  def __init__(self, args) -> None:
    super().__init__()

    self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
    kv_head_dim = args.embedding_size // self.n_kv_heads

    self.args = args
    self.num_rep = args.n_heads // self.n_kv_heads

    self.wq = nn.Linear(args.embedding_size, args.n_heads * args.embedding_size // args.n_heads, bias = False, dtype = torch.float16); '(C, C)'
    self.wk = nn.Linear(args.embedding_size, self.n_kv_heads*args.embedding_size // args.n_heads, bias = False, dtype = torch.float16); '(C, C // num_kv_head) ---> (C, kv_head_dim)'
    self.wv = nn.Linear(args.embedding_size, self.n_kv_heads*args.embedding_size // args.n_heads, bias = False, dtype = torch.float16); '(C, C // num_kv_head) ---> (C, kv_head_dim)'
    self.wo = nn.Linear(args.embedding_size, args.n_heads * args.embedding_size // args.n_heads, bias = False, dtype = torch.float16); '(C, C)'

    """ kv_head_dim = n_kv_heads*args.embedding_size // args.n_heads
    """

  def forward(self, x:torch.tensor, angles:torch.tensor, mask = True)->torch.tensor:
    """ x is a matrix of size (B, T, C). B=batch size, T= time step or sequence length, C = embedding dimension. x[0, 0, :] is the size of
    the one embedding vector of size C that is generated for only one token. So, x contains the embeddings vectors of B*T tokens.
    nh is the number of heads. n_kv_heads is the number of heads for keys and values. Here, n_kv_heads = 1.
    """
    B, T, C = x.shape

    qx = self.wq(x); '(B, T, C) @ (C, C) ----> (B, T, C) @ (B, C, C) ---> (B, T, C)'
    kx = self.wk(x); '(B, T, C) @ (C, n_kv_heads) ----> (B, T, C) @ (B, C, kv_head_dim) ---> (B, T, kv_head_dim) or (B, T, C // num_head)'
    vx = self.wv(x); '(B, T, C) @ (C, n_kv_heads) ----> (B, T, C) @ (B, C, kv_head_dim) ---> (B, T, kv_head_dim) or (B, T, C // num_head)'

    qx = qx.view(B, T, self.args.n_heads, C //  self.args.n_heads); '(B, T, nh, C/nh)'
    kx = kx.view(B, T, self.n_kv_heads, C //  self.args.n_heads); '(B, T, n_kv_heads, C/nh)'
    vx = vx.view(B, T, self.n_kv_heads, C //  self.args.n_heads); '(B, T, n_kv_heads, C/nh)'

    qx = apply_rotation(qx, angles); '(B, T, nh, C/nh)'
    kx = apply_rotation(kx, angles); '(B, T, n_kv_heads, C/nh)'


    qx = qx.transpose(1, 2); '(B, nh, T, C/nh)'
    kx = kx.transpose(1, 2); '(B, n_kv_heads, T, C/nh)'
    vx = vx.transpose(1, 2); '(B, n_kv_heads, T, C/nh)'

    if kx.shape[1] != 1: #n_kv_heads = 1, we do not need to repeat. That dimension will be broadcasted.
      kx = kx.repeat(1, self.num_rep, 1, 1); '(B, n_kv_heads, T, C/nh) ----> (B, nh, T, C/nh)'
      vx = vx.repeat(1, self.num_rep, 1, 1); '(B, n_kv_heads, T, C/nh) ----> (B, nh, T, C/nh)'

    weight = kx @ qx.transpose(-1, -2); '(B, nh, T, C/nh) @ (B, nh, C/nh, T) ----> (B, nh, T, C/nh) @ (B, nh, C/nh, T) ---> (B, nh, T, T)'

    if mask:
      mask = torch.full((T, T), float('-inf'), device = x.device).to(dtype = torch.float16)
      mask = torch.triu(mask, diagonal = 1)
      weight = weight + mask; '(B, nh, T, T) + (T, T) ---> (B, nh, T, T) + (B, nh, T, T) ----> (B, nh, T, T)'

    weight = torch.softmax(weight, dim = -1); '(B, nh, T, T)'
    out = weight @ vx; '(B, nh, T, T) @ (B, nh, T, C/nh) ----> (B, nh, T, T) @ (B, nh, T, C/nh) ---> (B, nh, T, C/nh)'
    out = out.transpose(1, 2); '(B, nh, T, C/nh) ----> (B, T, nh, C/nh)'
    out = out.contiguous().view(B, T, C); '(B, T, nh, C/nh) ----> (B, T, nh * C/nh) ----> (B, T, C)'
    return self.wo(out)

class FeedForward(nn.Module):
  def __init__(self, args)->None:
    super().__init__()

    hidden_dim = 4 * args.dim
    hidden_dim = int(2 * hidden_dim / 3)

    if args.ffn_dim_multiplier is not None:
      hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
    
    hidden = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
    
    self.w1 = nn.Linear(args.dim, hidden, bias = False, dtype = torch.float16)
    self.w2 = nn.Linear(hidden, args.dim, bias = False, dtype = torch.float16)
    self.w3 = nn.Linear(args.dim, hidden, bias = False, dtype = torch.float16)

  def forward(self, x)->torch.tensor:
    return self.w2(F.silu(self.w1(x)) * self.w3(x)); '(B, T, C)'
  
class TransformerBlock(nn.Module):
  def __init__(self, layer_id:int, args)->None:
    super().__init__()


    self.attention = GroupQueryAtention(args)
    self.feed_forward = FeedForward(args)
    
    self.layer_id = layer_id
    self.attention_norm = RMSNorm(args)
    self.ffn_norm = RMSNorm(args)

  def forward(self, x:torch.tensor, angles:torch.tensor)->torch.tensor:
    B, T, C = x.shape

    x = x + self.attention(self.attention_norm(x), angles)
    x = x + self.feed_forward(self.ffn_norm(x))

    return x

class Transformer(nn.Module):
  def __init__(self, args)->None:
    super().__init__()
    self.args = args
    self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim, dtype = torch.float16)
    self.layers = nn.ModuleList()

    for i in range(args.n_layers):
      self.layers.append(TransformerBlock(i, args))

    self.norm = RMSNorm(args)
    self.output = nn.Linear(args.dim, args.vocab_size, bias = False, dtype = torch.float16)

    self.angles = Compute_angles(args.dim // args.n_heads, args.max_seq_len, device = args.device)
    self.args = args

  def forward(self, x)->torch.tensor:
    B, T = x.shape
    device = x.device
    h = self.tok_embeddings(x)
    start_pos = 0
    self.angles = self.angles[start_pos : start_pos + self.args.max_seq_len]
    self.angles = self.angles.to(device)

    for i, layer in enumerate(self.layers):
      h = layer(h, self.angles)

    h = self.norm(h)
    h = self.output(h); '(B, T, vocab_size)'
    h = h.transpose(-1, -2); '(B, vocab_size, T)' #The cross entropy function takes input as (batch, classes, sequence)
    return h



    
