import torch
import os
import subprocess as sp
import torch.nn as nn


def print_size_of_model(model, label=""):
  print(f'model = {model}')
  
  path = './size_check'
  try:
    os.makedirs(path)
  except FileExistsError:
    print('File already exists')
        
  torch.save(model.state_dict(), path + "/temp.pt")
  size=os.path.getsize(path + "/temp.pt")
  print("model: ",label,' \t','Size (GB):', size/1e9)
  os.remove(path + '/temp.pt')
  return size


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values
  
  
def mark_only_lora_parameters_trainable(model:nn.Module, bias:str = 'none')->None:
  for n,p in model.named_parameters():
    if 'lora_' not in n:
      p.requires_grad = False

  if bias == 'none':
    return 

  elif bias == 'all':
    for n, p in model.named_parameters():
      if 'bias' in n:
        p.requires_grad = True

  elif bias == 'lora_only':
    for m in model.modules():
      print(f'm = {m}')
      if isinstance(m, LoraLayer) and hasattr(m, 'bias') and m.bias is not None:
        m.bias.requires_grad = True


  else:
    raise('NotImpementedError')