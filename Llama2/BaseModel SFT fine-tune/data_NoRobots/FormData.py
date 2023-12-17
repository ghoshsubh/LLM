
import pandas as pd
import sys
import numpy as np


tokenizer_path = './tokenizer.model'

sys.path.append('./FineTuneLLM/')
from tokenizer import Tokenizer

tokenizer = Tokenizer(model_path=tokenizer_path)


def form_data(file:str):
  data = pd.read_parquet(file)
  data = data['messages'].tolist()
  
  data_all =  '\n'.join(x['role'] + ':\n' + x['content'] for x1 in data for x in x1)
  
  x = np.array(tokenizer.encode(data_all, bos = False, eos = True))
  #print(x)
  return x


