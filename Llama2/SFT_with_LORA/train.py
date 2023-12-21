import os
import time
import math
import pickle
from contextlib import nullcontext
from tqdm import tqdm
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


import argparse
import parser_file
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple
import sys
import json
from transformer import Lora_Transformer
from utils import *

old_stdout = sys.stdout

log_file = open("./print_outputs.log","w")

sys.stdout = log_file

sys.path.append('./FineTuneLLM/data_NoRobots')
from FormData import form_data_no_robots

def ddp_setup():
  init_process_group(backend = "nccl")
  

class Trainer:
  def __init__(self,
               model:torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               train_data:None,
               val_data:None,
               loss_fn: torch.nn.CrossEntropyLoss,
               save_path:str,
               log_file:str,
               ags:argparse,
               use_amp:bool
               )->None:
    
    self.local_rank = int(os.environ["LOCAL_RANK"])
    self.global_rank = int(os.environ["RANK"])
    

    self.model = model.to(self.local_rank)
    
    
    self.optimizer = optimizer

    self.train_data = train_data
    self.val_data = val_data
    self.loss_fn = loss_fn
    self.save_path = save_path
    self.iters_run = 0
    
    self.use_amp = use_amp

    self.log_file = log_file
    self.ags = ags
    
    if os.path.exists(self.save_path+'/snapshot.pt'):

      self._load_snapshot(self.save_path+'/snapshot.pt')
      

    self.model = DDP(self.model, device_ids=[self.local_rank])    
  

  def _load_snapshot(self, snapshot_path):
    snapshot = torch.load(snapshot_path)
    self.model.load_state_dict(snapshot["MODEL_STATE"])
    self.iters_run = snapshot["Iter"]
    
    print(f"Resuming the model training from iterations {self.iters_run}")
    
  def _save_snapshot(self, Iter, name):
    snapshot = {}
    snapshot['MODEL_STATE'] = self.model.module.state_dict()
    snapshot['Iter'] = Iter
    snapshot['stats'] = self.stats
    snapshot['lr'] = self.updated_lr
    snapshot['optimizer'] = self.optimizer.state_dict()
    snapshot['ags'] = self.ags 
    snapshot['val_loss'] = self.val_loss1
    snapshot['train_loss'] = self.train_loss1
    snapshot['args'] = self.ags
    
    
    torch.save(snapshot, name)
    
    
  def train(self, max_iters):

    train_loss, val_loss = torch.tensor([0.0]), torch.tensor([0.0])
    
    if not os.path.exists(self.log_file + '/results.txt'):      
      text_file = open(self.log_file + '/results.txt', 'w')
    else:
      text_file = open(self.log_file + '/results.txt', 'a')
      
    best_train_loss, best_val_loss = 1e6, 1e6
    
    self.stats = {
      'batch_loss': [], 'train_loss': [], 'val_loss': []
    }
    
    self.scalar = torch.cuda.amp.GradScaler(enabled=self.use_amp)


    for e in tqdm(range(self.iters_run, max_iters)):
      self.model.train()
        
      loss = self._run_iter(self.ags)        
      
      if self.local_rank == 0 and (e+1) == max_iters:
        self._save_checkpoint(name = self.save_path +'/final_iter={}.pt'.format(e+1))
          
      if (e+1) % self.ags.save_every == 0:
        val_loss = self._estimate_loss('val', self.ags)
        train_loss = self._estimate_loss('train', self.ags)
                
        if self.local_rank == 0:
        
          self.stats['batch_loss'].append(loss)
          self.stats['train_loss'].append(train_loss)
          self.stats['val_loss'].append(val_loss)
          
          self.val_loss1, self.train_loss1 = val_loss, train_loss 
          self._save_snapshot(e+1, name = self.save_path + '/snapshot.pt') 

          if train_loss < best_train_loss:
            best_train_loss = train_loss
            self._save_snapshot(e+1, name = self.save_path + '/snapshot_train_loss.pt')  
            
          if val_loss < best_val_loss:
            
            best_val_loss = val_loss
            self._save_snapshot(e+1, name = self.save_path + '/snapshot_val_loss.pt')
          

      if self.local_rank == 0:
        line = f'Iteration:{e+1} | Train loss : {round(train_loss.item(), 4)} | Val loss : {round(val_loss.item(), 4)} | batch loss : {round(loss, 4)}\n'
        text_file.write(line)

    text_file.close()


  def _save_checkpoint(self, name):    
    torch.save(self.model.module.state_dict(), name)
    
    
  def _run_iter(self, ags):
    self.optimizer.zero_grad(set_to_none = True)
    
    source, output = self._get_batch('train', ags) 
    source, output = source.to(self.local_rank), output.to(self.local_rank)
    
    with torch.autocast(device_type = 'cuda', dtype = torch.float16, enabled = self.use_amp):
      logits = self.model(source)
      loss = self.loss_fn(logits, output)
      
      
    self.scalar.scale(loss).backward()
    self.scalar.step(self.optimizer)
    self.scalar.update()
    #if ags.grad_clip != 0.0:
    #  torch.nn.utils.clip_grad_norm_(self.model.parameters(), ags.grad_clip)
      
    return loss.item()

  
  @torch.no_grad()
  def _estimate_loss(self, split, ags):
    self.model.eval()
    losses = torch.zeros(ags.eval_iters)
    for k in range(ags.eval_iters):
      X, Y = self._get_batch(split, ags)
      X, Y = X.to(self.local_rank), Y.to(self.local_rank)
      logits = self.model(X)
      loss = self.loss_fn(logits, Y)
      losses[k] = loss.item()
    out = losses.mean()
    self.model.train()
    return out




  def _get_batch(self, split, ags):
    data = self.train_data if split == 'train' else self.val_data
    ix = torch.randint(len(data) - ags.max_seq_len, (ags.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+ags.max_seq_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+ags.max_seq_len]).astype(np.int64)) for i in ix])

    return x, y
  
  

def create_folder(ags):
  
  
  file_name = "/data={}/model_type={}/batch_size={}/block_size={}/max_iters={}/lr={}/save_every={}/dropout={}/bias={}/wd={}/beta1={}/\
    beta2={}/grad_clip={}/decay_lr={}/warmup_iters={}/lr_decay_iters={}/min_lr={}/".format(
              ags.data, ags.model_type, ags.batch_size, ags.max_seq_len, ags.max_iter, ags.lr, ags.save_every, ags.dropout, ags.bias, ags.wd, ags.beta1, ags.beta2,\
                ags.grad_clip, ags.decay_lr, ags.warmup_iters, ags.lr_decay_iters, ags.min_lr)
      
  file_final = './ALLMODELS/AllModels_1/final'+file_name
  log_file = './ALLMODELS/AllModels_1/logfile'+file_name
  
  if not os.path.exists(file_final):
    
    try:
      os.makedirs(file_final)
    except FileExistsError:
      print('Already exist the main file\n')

  if not os.path.exists(log_file):
    
    try:
      os.makedirs(log_file)
    except FileExistsError:
      print('Already exist the log file\n')

  return file_final, log_file, file_name


  
  
def load_train_objs(ModelArgs):
  model = Lora_Transformer(ModelArgs)
  optimizer = torch.optim.AdamW(model.parameters(), lr=ags.lr, betas=(ags.beta1, ags.beta2))

  return model, optimizer
  
      
def loss_fnc():
  return nn.CrossEntropyLoss()

def main(ags, train_data, val_data, file_final, log_file):
  
  ddp_setup()
  
  model, optimizer = load_train_objs(ags)

  mark_only_lora_parameters_trainable(model)
  
  if ags.finetune:
    print(f'Loading the base model')
    checkpoint_path = './AllBaseModels/llama-2-7b/consolidated.00.pth'
    checkpoint = torch.load(checkpoint_path, map_location = 'cpu')

    model.load_state_dict(checkpoint, strict = False)
  

    
  
  loss_fn = loss_fnc()


  trainer = Trainer(model, optimizer, train_data, val_data, loss_fn, save_path = file_final, log_file = log_file, ags = ags, use_amp = ags.use_amp)

  trainer.train(ags.max_iter)   
    
  destroy_process_group()

    
if __name__ == "__main__":


  parser = argparse.ArgumentParser(add_help=False)

  parser_from_file = parser_file.initialize(parser)

  ags = parser_from_file.parse_args()
  
  ags.finetune = True
  
  """Llama 7b base models hyper parameters
  {"dim": 4096, "multiple_of": 256, "n_heads": 32, "n_layers": 32, "norm_eps": 1e-05, "vocab_size": -1}

  """
  ags.dim = 4096
  ags.multiple_of = 256
  ags.n_heads = 32
  ags.n_layers = 32
  ags.norm_eps = 1e-5
  ags.vocab_size = 32000
  ags.embedding_size = ags.dim
  ags.n_kv_heads = None
  ags.ffn_dim_multiplier = None
  ags.batch_size = 1
  ags.max_seq_len = 2048
  ags.device = 'cpu'
  
  
  file_final, log_file, file_name = create_folder(ags)
  
  train_path = './data/train_sft-00000-of-00001-8aba5401a3b757f5.parquet'

  test_path = './data/test_sft-00000-of-00001-fe658ed8e3578d4a.parquet'

  if ags.data == 'no_robots':
    train_data = form_data_no_robots(train_path)
    val_data = form_data_no_robots(test_path)
  else:
    raise("Please give the correct data")
  
  print(f'Train data length = {len(train_data)}')
  print(f'Test data length = {len(val_data)}')

  main(ags, train_data, val_data, file_final, log_file)
  
    
  sys.stdout = old_stdout
  log_file.close()
