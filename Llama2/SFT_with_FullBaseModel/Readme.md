# Install
1. Clone the repository
```
git clone https://github.com/ghoshsubh/LLM.git
cd LLM/Llama2/SFT_with_FullBaseModel

```
2.Install the packages
```
conda env create -f environment.yml
source activate LLM
```

# Llamma 2 weights and tokenizer
Please go to this [website](https://ai.meta.com/llama/) and follow the process. Please download only base models.

# Memory requirements breakdown
```
Llamma-2 7B takes = 13.5 GB
```
We consider the batch size = 1 for the following calculation.

**Forward pass:** The network has total 32 layers. The first layer needs around 0.844 GB and each layer of other 31 layers needs 0.760 GB.
**Backward pass:** Each layer needs around 1 GB.
```
The minimum memory requirement = (13.5 + (1*0.844 + 31 * 0.760) + (32 * 1)) = 69.904 GB
```

# Requirements
You need at least `one 80 GB gpu` to train whole `llamma2-7B` base model with the fine-tune data set. I use `one H-100 80GB` gpu.

# Training
The following code will utilize only `one gpu`. use_amp = 0 indicates `no mixed precision`. As all tensors operations take place in `torch.float16`, we do not need mixed precision concept. 
```
torchrun --standalone --nproc_per_node=1 train.py --use_amp 0 --data no_robots

```
The following code will utilizes `all gpus` in a node, but you may see `out-of-memory` error. When we wrap the model with DDP, it eats a huge memory that leads to the error. 
```
torchrun --standalone --nproc_per_node=gpu train.py --use_amp 0 --data no_robots

```

