#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比测试脚本 - 微调模型 vs 原始模型
用于比较微调前后的情感分析效果
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_models():
    """
    加载原始模型和微调后的模型
    """
    BASE_MODEL_PATH = "./qwen/Qwen2_5-1_5B-Instruct"
    LORA_MODEL_PATH = "output_checkpoints/checkpoint-688"

    print("=" * 80)
    print("模型对比测试 - 微调前 vs 微调后")
    print("=" * 80)

    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    print("\n正在加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 加载原始模型
    print(f"正在加载原始模型: {BASE_MODEL_PATH}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 加载微调后的模型
    print(f"正在加载微调后的模型: {LORA_MODEL_PATH}")
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    finetuned_model = PeftModel.from_pretrained(finetuned_model, LORA_MODEL_PATH)

    print("✓ 所有模型加载完成！\n")

    return base_model, finetuned_model, tokenizer

def predict_sentiment(model, tokenizer, text, max_new_tokens=80):
    """
    预测单条文本的情感
    """
    messages = [
        {
            "role": "system",
            "content": "你是一位情感分析专家。请分析以下文本的情感倾向，并输出最准确的情感标签。"
        },
        {
            "role": "user",
            "content": text
        }
    ]

    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    return response

def parse_label(response):
    """
    从模型输出中解析情感标签
    """
    try:
        if "{" in response and "}" in response:
            json_str = response[response.find("{"):response.rfind("}") + 1]
            result = json.loads(json_str)
            return result.get("label", "未知")
        else:
            return response.strip()
    except:
        return "解析失败"

def compare_models(base_model, finetuned_model, tokenizer, test_cases):
    """
    对比两个模型的预测结果
    """
    print("=" * 80)
    print("开始对比测试")
    print("=" * 80 + "\n")

    correct_base = 0
    correct_finetuned = 0

    for i, (text, expected_label) in enumerate(test_cases, 1):
        print(f"【测试 {i}/{len(test_cases)}】")
        print(f"输入文本: {text}")
        print(f"期望标签: {expected_label}")

        # 原始模型预测
        print("\n--- 原始模型 ---")
        base_response = predict_sentiment(base_model, tokenizer, text)
        base_label = parse_label(base_response)
        print(f"输出: {base_response}")
        print(f"预测标签: {base_label}")

        base_correct = (expected_label in base_label)
        if base_correct:
            correct_base += 1
            print("✓ 正确")
        else:
            print("✗ 错误")

        # 微调模型预测
        print("\n--- 微调模型 ---")
        finetuned_response = predict_sentiment(finetuned_model, tokenizer, text)
        finetuned_label = parse_label(finetuned_response)
        print(f"输出: {finetuned_response}")
        print(f"预测标签: {finetuned_label}")

        finetuned_correct = (expected_label in finetuned_label)
        if finetuned_correct:
            correct_finetuned += 1
            print("✓ 正确")
        else:
            print("✗ 错误")

        # 对比结果
        print("\n--- 对比总结 ---")
        if base_correct and finetuned_correct:
            print("两个模型都正确")
        elif base_correct and not finetuned_correct:
            print("原始模型正确，微调模型错误 ⚠️")
        elif not base_correct and finetuned_correct:
            print("微调模型正确，原始模型错误 ✅")
        else:
            print("两个模型都错误")

        print("-" * 80 + "\n")

    # 计算准确率
    total = len(test_cases)
    base_accuracy = correct_base / total * 100
    finetuned_accuracy = correct_finetuned / total * 100
    improvement = finetuned_accuracy - base_accuracy

    print("\n" + "=" * 80)
    print("总体对比结果")
    print("=" * 80)
    print(f"原始模型准确率: {correct_base}/{total} ({base_accuracy:.1f}%)")
    print(f"微调模型准确率: {correct_finetuned}/{total} ({finetuned_accuracy:.1f}%)")
    print(f"提升幅度: {improvement:+.1f}%")
    print("=" * 80 + "\n")

def load_test_samples(file_path, num_samples=50):
    """
    从测试数据中加载样本
    """
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            data = json.loads(line.strip())
            user_content = None
            true_label = None
            for msg in data['messages']:
                if msg['role'] == 'user':
                    user_content = msg['content']
                elif msg['role'] == 'assistant':
                    try:
                        result = json.loads(msg['content'])
                        true_label = result.get('label', '未知')
                    except:
                        pass
            if user_content and true_label:
                samples.append((user_content, true_label))
    return samples

def main():
    # 加载模型
    base_model, finetuned_model, tokenizer = load_models()

    # 测试用例：包含多种情感类别
    test_cases = [
        # 正面评价
        ("很好吃，送餐也很快！", "满意/愉快"),
        ("服务态度特别好，非常满意！", "满意/愉快"),
        ("味道鲜美，分量足，值得推荐！", "满意/愉快"),
        ("性价比很高，推荐！", "满意/愉快"),
        ("感觉挺不错的！", "满意/愉快"),
        ("非常快，态度好。", "满意/愉快"),
        ("方便，快捷，味道可口，快递给力", "满意/愉快"),
        ("菜味道很棒！送餐很及时！", "满意/愉快"),
        ("真心不错，吃了一次就爱上了", "满意/愉快"),
        ("五星好评，强烈推荐！", "满意/愉快"),

        # 负面评价
        ("味道太差了，再也不来了！", "失望/愤怒"),
        ("等了两个小时才送到，饭菜都凉了！", "失望/愤怒"),
        ("太难吃了，浪费钱！", "失望/愤怒"),
        ("送餐太慢了，体验很差！", "失望/愤怒"),
        ("品味一般，服务差！", "失望/愤怒"),
        ("送来都冰凉了", "失望/愤怒"),
        ("两个多小时都没送过来", "失望/愤怒"),
        ("送的是在是太慢了", "失望/愤怒"),
        ("真心太慢了,我等了两个小时", "失望/愤怒"),
        ("送餐不守时，太难吃。", "失望/愤怒"),

        # 中性/混合评价
        ("还可以吧，没有什么特别的。", "中性"),
        ("一般般，不好不坏。", "中性"),
        ("没什么味儿～～", "中性"),
        ("味道还行，比较辣", "中性"),
        ("第一，这个系统怎么回事？", "中性"),
        ("味道不错，速度太慢", "失望/愤怒"),
        ("菜做的很棒，可是送餐太慢了", "失望/愤怒"),
        ("味道很好，就是送餐太慢了。", "失望/愤怒"),
        ("东西挺好，有点儿辣有点儿咸。送餐速度慢哭了", "失望/愤怒"),
        ("等了一个半小时才吃到了午饭。牛肉还不错，芹菜老了。", "失望/愤怒"),

        # 复杂场景
        ("送餐慢，10点40下单，12点42送到，今天再试试", "失望/愤怒"),
        ("之前送的很快，今天不知道为什么特别慢", "失望/愤怒"),
        ("比预计的送达时间晚了半个多小时", "失望/愤怒"),
        ("味道很好，服务态度也不错", "满意/愉快"),
        ("整体体验不错，还会再来的", "满意/愉快"),
        ("第一次订餐，感觉还可以", "中性"),
        ("送餐有点慢，但味道还行", "中性"),
    ]

    print(f"测试用例数量: {len(test_cases)}")
    print("包含: 正面评价、负面评价、中性/混合评价\n")

    # 运行对比测试
    compare_models(base_model, finetuned_model, tokenizer, test_cases)

    # 额外：使用测试数据集进行评估
    if os.path.exists("data/dev.jsonl"):
        print("\n" + "=" * 80)
        print("使用测试数据集进行评估")
        print("=" * 80 + "\n")

        dataset_samples = load_test_samples("data/dev.jsonl", num_samples=30)
        print(f"从测试数据集加载了 {len(dataset_samples)} 个样本\n")

        compare_models(base_model, finetuned_model, tokenizer, dataset_samples)

    print("\n所有测试完成！")

if __name__ == "__main__":
    main()
