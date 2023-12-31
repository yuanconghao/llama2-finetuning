## create dataset

The main fine-tuning script is written in a general format that would require you to provide a `jsonl` file for train and test datasets in addition to a json file listing the special tokens used in your dataset.
For example each row in your dataset might be formated like the following:
```json
{"input": "<ASSISTANT>How can I help you?</ASSISTANT><USER>how is the weather?</USER>"}
```
And the special tokens can be:

```json
{"tokens": ["<ASSISTANT>", "</ASSISTANT>", "<USER>", "</USER>"]}
```

### usage
[create_dataset.py][6]
```shell
python create_dataset.py
```

## convert llama weights to transformers weights

convert llama weights to transformers weights and upload to huggingface

### convert
[convert_llama_weights_to_hf.py][1]
```shell
python convert_llama_weights_to_hf.py \
  --input_dir /home/work/virtual-venv/lora-env/data/Llama2 \
  --model_size 7B  \ 
  --output_dir /home/work/virtual-venv/lora-env/data/llama2-chat-hf
```
### hf model

[conghao/llama2-7b-chat-hf][5]

## llama2 fine tuning 

### method

[peft fine tuning][2]

### qlora fine tuning

[qlora_finetuning.py][3]

```shell
python qlora_finetuning.py
#nohup python qlora_finetuning.py 2>/tmp/llama2-lora.log 1>&2 &
```

fine tuning result: [conghao/llama2-qlora-finetuning][4]


## llama2 inference

[inference_llama2.py][7]

```shell
python inference_llama2.py
```


[1]:./convert_llama_weights_to_hf.py
[2]:https://deeplearner.top/2023/08/24/AIGC-%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%BE%AE%E8%B0%83-PEFT%E6%8A%80%E6%9C%AF%E7%AE%80%E4%BB%8B/
[3]:./qlora_finetuning.py
[4]:https://huggingface.co/conghao/llama2-qlora-finetuning
[5]:https://huggingface.co/conghao/llama2-7b-chat-hf/tree/main
[6]:./create_dataset.py
[7]:./inference_llama2.py