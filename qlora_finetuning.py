# -*- coding: utf-8 -*-

"""
pip install
"""
#pip install -r requirements.txt

"""Load Dataset"""

from datasets import load_dataset, Dataset
import pandas as pd

# huggingface dataset
#dataset_name = "Photolens/MedText-llama-2"
#dataset = load_dataset(dataset_name, split="train")
#print(dataset)
#print(dataset['text'][0])

# local dataset
def local_dataset(dataset_name):
    if dataset_name.endswith('.json') or dataset_name.endswith('.jsonl'):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    #split_dataset = full_dataset.train_test_split(test_size=0.1)
    return full_dataset

dataset_name = "./data_med/train.jsonl"
dataset = local_dataset(dataset_name)
print(dataset)


"""Load Model"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

#huggingface model
# model_name = "conghao/llama2-7b-chat-hf"
#local model
model_name = "/home/work/virtual-venv/lora-env/data/llama2-chat-hf"

# 量化config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 权重被量化为4位
    bnb_4bit_quant_type="nf4",  # nft4类型
    bnb_4bit_compute_dtype=torch.float16,
)

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained(
    model_name, # 模型路径
    quantization_config=bnb_config, # 量化配置
    trust_remote_code=True  # 是否信任远程代码
)

model.config.use_cache = False

"""Tokenizer"""

# 预训练分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

"""Lora Config"""

from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer

# Lora Config
lora_alpha = 16 # LoRA低秩矩阵的缩放系数，为一个常数超参，调整alpha与调整学习率类似。
lora_dropout = 0.1  # LoRA 层的丢弃（dropout）率，取值范围为[0, 1)
lora_r = 64 # LoRA低秩矩阵的维数。关于秩的选择，通常，使用4，8，16即可。
task_type = 'CAUSAL_LM' # 指定任务类型。如条件生成任务（SEQ_2_SEQ_LM），因果语言建模（CAUSAL_LM）等。

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type=task_type,
)

# Trainer arguments
output_dir = "./results"    # 指定保存模型和日志的目录。
per_device_train_batch_size = 8 # 对于每个设备（例如每块GPU），在训练期间每次迭代使用的样本数量。
gradient_accumulation_steps = 4 # 如前所述，它决定了在执行模型参数更新之前要累积的梯度步数。
optim = "paged_adamw_32bit" # 使用的优化器名称。在这里，它看起来是一个特定的32位版本的AdamW优化器。
save_steps = 100    # 每训练多少步后保存模型的频率。
logging_steps = 10  # 每训练多少步后记录日志的频率。
learning_rate = 2e-5    # 模型训练的学习率。在此代码中，学习率已经被调低（之前为2e-4，现在为2e-5），这意味着模型训练的“步长”会更小，从而可能提供更精确、更平滑的权重调整。
max_grad_norm = 0.3 # 梯度裁剪的阈值，它可以防止梯度爆炸问题。
max_steps = 1000 # 总共训练的步数。
warmup_ratio = 0.03 # 学习率预热的比例。预热是训练开始时逐渐增加学习率的过程，它可以帮助模型更稳定地收敛。
lr_scheduler_type = "constant"  # 学习率调度器的类型。在这里，它被设置为"constant"，这意味着学习率在整个训练过程中保持不变。
#lr_scheduler_type = 'linear'

traning_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
)

# Load trainer
max_seq_length = 512    # 定义了模型接受的序列的最大长度，这里设置为512。这是Transformer模型常用的长度，如BERT和GPT-2等。

# SFTTrainer训练器
trainer = SFTTrainer(
    model=model,    # 要训练的模型实例
    train_dataset=dataset,  # 用于训练的数据集
    peft_config=peft_config,    # Lora的配置，可能与模型的某种特性或训练策略有关。
    dataset_text_field="input",  # 指定了数据集中用作输入的字段的名称，这里设置为"text"
    max_seq_length=max_seq_length,  # 指定了模型接受的序列的最大长度
    tokenizer=tokenizer,    # 预训练分词器，用于将文本转换为模型可以理解的格式。
    args=traning_arguments, # 训练参数，如学习率、批次大小等
)

for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)

"""Start train"""

trainer.train()

"""Save Model"""

model_to_save = trainer.model.module() if hasattr(trainer.model, 'module') else trainer.model
model_to_save.save_pretrained("outputs")

lora_config = LoraConfig.from_pretrained("outputs")
model = get_peft_model(model, lora_config)  # 使用先前加载的LoRA配置，对模型进行一些修改或更新

print(dataset['text'][0])
text = dataset['text'][0]
text = text.split("[/INST]")
input_text = text[0].replace("<s>[INST]", "").strip()
output_text = text[1].strip()
print("--------dataset input--------")
print(input_text)
print("--------dataset output--------")
print(output_text)

"""Test"""

device = "cuda:0"
print("--------model input--------")
print(input_text)
inputs = tokenizer(input_text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=512)
print("--------model output--------")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

"""Upload to Huggingface"""

from huggingface_hub import login

hf_token = "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"
login(hf_token)

model.push_to_hub("llama2-qlora-med")
