# Open Access data sets for SFT(supervised fine tuning) 
There are theree categories of data sets:

1. **Dialog:** Each entry contains continous conversation
2.  **Pairs:** Each entry is an input-output pair
3.  **Context:** Each Entry has a context text and related QA pairs.


| Data set name | Released data | Data type | Size(Train+Test) | Description|
| :--- | :------: | ----: | ---: | ---: |
| [no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots)        |   11/23   | SFT |9.5k + 0.5k|High quality human created SFT data set.|
|     [function_calling_extended](https://huggingface.co/datasets/Trelis/function_calling_extended) |08/23 | Pairs||High quality human created dataset from enhance LM's API using ability.|
| [Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus)    |  08/23   | Pairs |24.9k+0.1k|A very high quality dataset for improving LM's STEM reasoning ability.|
| [WebGLM-qa](https://huggingface.co/datasets/THUDM/webglm-qa/viewer/default/train) | 07/23  | Pairs |43.6k+1k|Dataset used by WebGLM, which is a QA system based on LLM and Internet. Each of the entry in this dataset comprise a question, a response and a reference. The response is grounded in the reference.|

# Install
1. Clone the repository
```
git clone https://github.com/ghoshsubh/LLM.git
cd LLM/Llama2/BaseModel SFT fine-tune

```
2.Install the packages
```
conda env create -f environment.yml
source activate LLM
```

# Llamma 2 weights and tokenizer
Please go to this [website](https://ai.meta.com/llama/) and follow the process. Please download only base models.

# Requirements
You need at least `one 80 GB gpu` to train whole `llamma2-7B` base model with the fine-tune data set. I use `one H-100 80GB` gpu.

# Training
The following code will utilize only `one gpu`. use_amp == 0 indicates `no mixed precision`. As all tensors operations take place in `torch.float16`, we do not need mixed precision concept. 
```
torchrun --standalone --nproc_per_node=1 train.py --use_amp 0 --data no_robots

```
The following code will utilizes `all gpus` in a node, but you may see `out-of-memory` error. When we wrap the model with DDP, it eats a huge memory that leads to the error. 
```
torchrun --standalone --nproc_per_node=gpu train.py --use_amp 0 --data no_robots

```

