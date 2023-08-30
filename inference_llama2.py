from transformers import AutoModelForCausalLM, LlamaTokenizer
import torch

model_base_id = "/home/work/virtual-venv/lora-env/data/llama2-chat-hf"
model_base = AutoModelForCausalLM.from_pretrained(model_base_id, load_in_4bit=True)

tokenizer = LlamaTokenizer.from_pretrained(model_base_id)

# print(model)

device = torch.device("cuda:0")

# model = model.to(device)

print("\n------------------------desc------------------------")

text = "Hello, my name is "
inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model_base.generate(**inputs, max_new_tokens=20, do_sample=True, top_k=30, top_p=0.85)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("\n\nInput: ")
line = input()
while line:
    print("\n-----------------------base model-------------------------\n")
    inputs = tokenizer(line, return_tensors="pt").to(device)
    outputs_base = model_base.generate(**inputs, max_new_tokens=512, do_sample=True, top_k=30, top_p=0.85)
    print("Output: ", tokenizer.decode(outputs_base[0], skip_special_tokens=True))
    print("\n\nInput: ")
    line = input()
