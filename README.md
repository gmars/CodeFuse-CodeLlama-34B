# Model Card for CodeFuse-CodeLlama-34B
<p align="left">
  <img src="./LOGO.png" width="100%" />
</p>

[[中文]](#chinese)    [[English]](#english)



<a id="english"></a>

## Model Description

CodeFuse-CodeLlama-34B is a 34B Code-LLM finetuned by QLoRA of multiple code tasks（600k instrunctions/answers） on the base model CodeLlama-34b-Python. 
The context length of finetuning is 4K while it is able to be finetuned by 16k context if necessary.
<br>

## News and Updates

🔥🔥🔥 CodeFuse-CodeLlama34B-MFT has achived 74.4% of pass@1 on HumanEval, which is SOTA at present.

<br>

## Performance

| Model                         | HumanEval(pass@1) |
| :---------------------------- | :---------------: |
| CodeLlama-34b                 |   48.8%(greedy decoding)   |
| CodeLlama-34b-Python          |   53.7%(greedy decoding)   |
| **CodeFuse-CodeLlama-34B** | **74.4%**(greedy decoding) |

<br>

## Requirements

* python>=3.8 
* pytorch>=2.0.0
* transformers==4.32.0
* Sentencepiece
* CUDA 11.4
  <br>

##  Inference String Format

The inference string is a concatenated string formed by combining conversation data(system, human and bot contents) in the training data format.  It is used as input during the inference process.
Here is an example format of the concatenated string:

```python
"""
<|role_start|>system<|role_end|>System instruction
<|role_start|>human<|role_end|>Human 1st round input
<|role_start|>bot<|role_end|>Bot 1st round output</s>
<|role_start|>human<|role_end|>Human 2nd round input
<|role_start|>bot<|role_end|>Bot 2nd round output</s>
...
...
...
<|role_start|>human<|role_end|>Human nth round input
<|role_start|>bot<|role_end|>{Bot output to be genreated}</s>
"""
```

When applying inference, you always make your input string end with "<|role_start|>bot<|role_end|>" to ask the model generating answers.

## Quickstart

```bash
pip install -r requirements.txt
```

```python
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
)
tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, trust_remote_code=True, use_fast=False, legacy=False)
tokenizer.padding_side = "left"
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<unk>")
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, trust_remote_code=True)

HUMAN_ROLE_START_TAG = "<|role_start|>human<|role_end|>"
BOT_ROLE_START_TAG = "<|role_start|>bot<|role_end|>"

text = f"{HUMAN_ROLE_START_TAG}write a python function of quick sort.{BOT_ROLE_START_TAG}" 
inputs = tokenizer(text, return_tensors='pt', padding=True, add_special_tokens=False).to("cuda")
outputs = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=512,
        top_p=0.95,
        temperature=0.1,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
gen_text = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(gen_text)
```













<a id="chinese"></a>

## 模型简介

CodeFuse-CodeLlama34B-MFT 是一个通过QLoRA对基座模型CodeLlama-34b-Python进行多代码任务微调的代码大模型。模型微调采用了4k上下文。如果有必要，可以扩展到16k。
<br>

## 新闻

🔥🔥🔥 CodeFuse-CodeLlama34B-MFT模型在HumanEval pass@1上可以达到74.4%, 为当前开源SOTA。

<br>

## 评测表现(代码)

| 模型                         | HumanEval(pass@1) |
| :---------------------------- | :---------------: |
| CodeLlama-34b                 |   48.8%(greedy decoding)   |
| CodeLlama-34b-Python          |   53.7%(greedy decoding)   |
| **CodeFuse-CodeLlama-34B** | **74.4%**(greedy decoding) |
<br>

## Requirements

* python>=3.8 
* pytorch>=2.0.0
* transformers==4.32.0
* CUDA 11.4
<br>

## 推理数据格式

推理数据为模型在训练数据格式下拼接的字符串形式，它也是推理时输入prompt拼接的方式：

```python
"""
<|role_start|>system<|role_end|>这是System指令
<|role_start|>human<|role_end|>这是第1轮用户输入的问题
<|role_start|>bot<|role_end|>这是第1轮模型生成的内容</s>
<|role_start|>human<|role_end|>这是第2轮用户输入的问题
<|role_start|>bot<|role_end|>这是第2轮模型生成的内容</s>
...
...
...
<|role_start|>human<|role_end|>这是第n轮用户输入的问题
<|role_start|>bot<|role_end|>{模型现在要生成的内容}</s>
"""
```

推理时，请确保拼接的prompt字符串以"<|role_start|>bot<|role_end|>"结尾，引导模型生成回答。

## 快速使用

```python
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
)
tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, trust_remote_code=True, use_fast=False, legacy=False)
tokenizer.padding_side = "left"
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<unk>")
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, trust_remote_code=True)

HUMAN_ROLE_START_TAG = "<|role_start|>human<|role_end|>"
BOT_ROLE_START_TAG = "<|role_start|>bot<|role_end|>"

text = f"{HUMAN_ROLE_START_TAG}write a python function of quick sort.{BOT_ROLE_START_TAG}" 
inputs = tokenizer(text, return_tensors='pt', padding=True, add_special_tokens=False).to("cuda")
outputs = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=512,
        top_p=0.95,
        temperature=0.1,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
gen_text = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(gen_text)
```


