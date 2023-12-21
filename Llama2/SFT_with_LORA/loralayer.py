import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from typing import Optional, List

class LoraLayer():
  def __init__(self, r:int, lora_alpha:int, lora_dropout:float, merge_weights:bool)->None:

    self.r = r
    self.lora_alpha = lora_alpha
    if lora_dropout > 0:
      self.lora_dropout == nn.Dropout(p = lora_dropout)
    else:
      self.lora_dropout = lambda x:x

    self.merged = False
    self.merge_weights = merge_weights



class Lora_Embedding(nn.Embedding, LoraLayer):
  def __init__(self, num_embeddings:int, embedding_dim:int, r:int = 0,  lora_alpha:int = 1, merge_weights:bool = True, **kwargs):

    nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
    LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0, merge_weights=merge_weights)

    if r > 0:
      self.lora_A = nn.Parameter(self.weight.new_zeros(r, num_embeddings))
      self.lora_B = nn.Parameter(self.weight.new_zeros(embedding_dim, r))

      self.scaling = self.lora_alpha / self.r

      self.weight.requires_grad = False

    self.reset_parameters()


  def reset_parameters(self) -> None:
    nn.Embedding.reset_parameters(self)
    if hasattr(self, 'lora_A'):
      nn.init.zeros_(self.lora_A)
      nn.init.normal_(self.lora_B)

  def train(self, mode:bool = True):
    nn.Embedding.train(self, mode)

    if mode:
      if self.merged_weights and self.merged:
        if self.r > 0:
          self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
        self.merged = False

    else:
      if self.merged_weights and not self.merged:
        if self.r > 0:
          self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
        self.merged = True


  def forward(self, x:torch.tensor):
    if self.r > 0 and not self.merged:
      results = nn.Embedding.forward(self, x)
      print(results.shape)
      after_A = F.embedding(
          x, self.lora_A.transpose(0, 1), self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse
      )
      results += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
      return results
    else:
      return nn.Embedding.forward(self, x)




class Lora_Linear(nn.Linear, LoraLayer):

  def __init__(self, in_features: int, out_features: int, r:int, lora_alpha:int, lora_dropout:float, fan_in_fan_out:bool = False, merge_weights:bool = True, **kwargs) -> None:

    nn.Linear.__init__(self, in_features, out_features, **kwargs)
    LoraLayer.__init__(self, r = r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

    self.fan_in_fan_out = fan_in_fan_out
    self.in_features = in_features

    if r > 0:
      self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
      self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))

      self.scaling = self.lora_alpha / self.r

      self.weight.requires_grad = False

    self.reset_parameters()

    if fan_in_fan_out:
      self.weight.data = self.weight.data.transpose(0, 1)

  def reset_parameters(self) -> None:
    nn.Linear.reset_parameters(self)
    if hasattr(self, 'lora_A'):
      nn.init.kaiming_uniform_(self.lora_A, a = math.sqrt(self.in_features))
      nn.init.zeros_(self.lora_B)

  def train(self, mode:bool = True):
    def T(w):
      return w.transpose(0, 1) if self.fan_in_fan_out else w
    nn.Linear.train(self, mode)

    if mode:
      if self.merge_weights and self.merged:
        if self.r > 0:
          self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling

        self.merged = False

    else:
      if self.merge_weights and not self.merged:
        if self.r > 0:
          self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
        self.merged = False

  def forward(self, x:torch.tensor):
    def T(w):
      return w.transpose(0, 1) if self.fan_in_fan_out else w

    if self.r > 0 and not self.merged:
      results = F.linear(x, T(self.weight), bias = self.bias)

      results += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
      return results

    else:
      return F.linear(x, T(self.weight), bias = self.bias)



