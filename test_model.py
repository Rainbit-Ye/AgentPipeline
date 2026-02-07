#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速测试脚本 - 验证微调后的模型
"""

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    # 配置路径
    BASE_MODEL_PATH = "/home/user1/lioduanye/AgentPipeline/qwen/Qwen2_5-1_5B-Instruct"
    LORA_MODEL_PATH = "output_checkpoints/checkpoint-688"

    print("=" * 80)
    print("快速测试微调后的情感分析模型")
    print("=" * 80)

    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    print("\n步骤1: 加载模型和tokenizer...")
    print(f"基础模型: {BASE_MODEL_PATH}")
    print(f"LoRA权重: {LORA_MODEL_PATH}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
    print("✓ 模型加载完成！\n")

    # 测试样例
    test_cases = [
        "很好吃，送餐也很快！",
        "味道太差了，再也不来了！",
        "还可以吧，没有什么特别的。",
        "服务态度特别好，非常满意！",
        "等了两个小时才送到，饭菜都凉了！"
    ]

    print("步骤2: 运行测试...\n")

    for i, text in enumerate(test_cases, 1):
        print(f"【测试 {i}/{len(test_cases)}】")
        print(f"输入: {text}")

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
                max_new_tokens=80,
                temperature=0.1,
                do_sample=True
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        print(f"输出: {response}")

        try:
            if "{" in response:
                json_str = response[response.find("{"):response.rfind("}") + 1]
                result = json.loads(json_str)
                label = result.get("label", "未知")
                print(f"情感标签: {label}")
            else:
                print(f"情感标签: {response.strip()}")
        except:
            print(f"情感标签: 解析失败")

        print("-" * 80 + "\n")

    print("=" * 80)
    print("测试完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
