#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版模型对比测试
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

BASE_MODEL_PATH = "./qwen/Qwen2_5-1_5B-Instruct"
LORA_MODEL_PATH = "output_checkpoints/checkpoint-688"

print("正在加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

print("正在加载原始模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print("正在加载微调模型...")
finetuned_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
finetuned_model = PeftModel.from_pretrained(finetuned_model, LORA_MODEL_PATH)

print("✓ 模型加载完成！\n")

test_cases = [
    ("很好吃，送餐也很快！", "满意/愉快"),
    ("味道太差了，再也不来了！", "失望/愤怒"),
    ("还可以吧，没有什么特别的。", "中性"),
    ("服务态度特别好，非常满意！", "满意/愉快"),
    ("等了两个小时才送到，饭菜都凉了！", "失望/愤怒"),
    ("送餐很慢，但味道不错", "中性"),
    ("味道鲜美，分量足，值得推荐！", "满意/愉快"),
    ("送餐太慢了，体验很差！", "失望/愤怒"),
    ("真心不错，吃了一次就爱上了", "满意/愉快"),
    ("送餐不守时，太难吃。", "失望/愤怒"),
]

def predict(model, text):
    messages = [
        {"role": "system", "content": "你是一位情感分析专家。请分析以下文本的情感倾向，并输出最准确的情感标签。"},
        {"role": "user", "content": text}
    ]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=60, temperature=0.1, do_sample=True)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response

def parse(response):
    try:
        if "{" in response and "}" in response:
            json_str = response[response.find("{"):response.rfind("}") + 1]
            result = json.loads(json_str)
            return result.get("label", "未知")
        return response.strip()
    except:
        return "解析失败"

print("=" * 80)
print("模型对比测试结果")
print("=" * 80)

correct_base = 0
correct_finetuned = 0

for i, (text, expected) in enumerate(test_cases, 1):
    print(f"\n【测试 {i}/{len(test_cases)}】")
    print(f"输入: {text}")
    print(f"期望: {expected}")
    
    base_resp = predict(base_model, text)
    base_label = parse(base_resp)
    print(f"原始模型: {base_label}")
    
    finetuned_resp = predict(finetuned_model, text)
    finetuned_label = parse(finetuned_resp)
    print(f"微调模型: {finetuned_label}")
    
    base_correct = expected in base_label
    finetuned_correct = expected in finetuned_label
    
    if base_correct:
        correct_base += 1
        print("原始模型: ✓")
    else:
        print("原始模型: ✗")
    
    if finetuned_correct:
        correct_finetuned += 1
        print("微调模型: ✓")
    else:
        print("微调模型: ✗")
    
    if base_correct and not finetuned_correct:
        print("原始模型更好")
    elif not base_correct and finetuned_correct:
        print("微调模型更好")
    print("-" * 60)

print("\n" + "=" * 80)
print("总体结果")
print("=" * 80)
print(f"原始模型准确率: {correct_base}/{len(test_cases)} ({correct_base/len(test_cases)*100:.1f}%)")
print(f"微调模型准确率: {correct_finetuned}/{len(test_cases)} ({correct_finetuned/len(test_cases)*100:.1f}%)")
print(f"提升: {(correct_finetuned-correct_base)/len(test_cases)*100:+.1f}%")
print("=" * 80)
